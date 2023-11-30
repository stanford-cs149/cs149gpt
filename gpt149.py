import argparse
import time
import math
import random
import inspect
from dataclasses import dataclass
import sys, getopt
from os import getcwd, path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
import module_ref as ms

NUM_THREADS=8
torch.set_num_threads(NUM_THREADS)

ispc_path = getcwd() + "/module_ispc.o"
if not path.exists(ispc_path): ispc_path = ""

print("\nCompiling code into a PyTorch module...\n\n")
mr = load(name="custom_module", sources=["module.cpp"],  extra_cflags=["-mavx", "-O3", "-fopenmp"], extra_ldflags=[ispc_path])
correctness_error_message = "\n-------------------------------------------\n YOUR ATTENTION PRODUCED INCORRECT RESULTS"

class CustomAttention(nn.Module):
    def __init__(self, Q,K,V, B, H, N, d, isRef=False, bc=256, br=256):
        super(nn.Module, self).__init__()
        self.Q=Q
        self.K=K
        self.V=V
        self.bc=bc
        self.br=br
        self.B=B
        self.H=H
        self.N=N
        self.d=d
        self.isRef=isRef

    #part 1
    def myUnfusedAttention(self):
        if self.isRef:
            with record_function("STUDENT - NAIVE ATTENTION"):
                temp = torch.zeros((self.N, self.N))
                out = mr.myNaiveAttention(self.Q, self.K, self.V, temp, self.B, self.H, self.N, self.d)
            return out
        with record_function("REFERENCE - NAIVE ATTENTION"):
            temp = torch.zeros((self.N, self.N))
            out = ms.myNaiveAttention(self.Q, self.K, self.V, temp, self.B, self.H, self.N, self.d)
        return out

    #part 2
    def myUnfusedAttentionBlocked(self):
        if self.isRef:
            with record_function("STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX"):
                temp = torch.zeros((self.N, self.N))
                out = mr.myUnfusedAttentionBlocked(self.Q, self.K, self.V, temp, self.B, self.H, self.N, self.d)
            return out 
        with record_function("REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX"):
            temp = torch.zeros((self.N, self.N))
            out = ms.myUnfusedAttentionBlocked(self.Q, self.K, self.V, temp, self.B, self.H, self.N, self.d)
        return out

    #part 3
    def myFusedAttention(self):
        if self.isRef:
            with record_function("STUDENT - FUSED ATTENTION"):
                temp = torch.zeros((NUM_THREADS, self.N))
                out = mr.myFusedAttention(self.Q, self.K, self.V, temp, self.B, self.H, self.N, self.d)
            return out
        with record_function("REFERENCE - FUSED ATTENTION"):
            temp = torch.zeros((NUM_THREADS, self.N))
            out = ms.myFusedAttention(self.Q, self.K, self.V, temp, self.B, self.H, self.N, self.d)
        return out

    #part 4
    def myFlashAttention(self):
        d = self.d
        Qi = torch.zeros((self.br, self.d))
        Kj = torch.zeros((self.bc, self.d))
        Vj = torch.zeros((self.bc, self.d))
        Sij = torch.zeros((self.br, self.bc))
        Pij = torch.zeros((self.br, self.bc))
        PV = torch.zeros((self.br, d))
        Oi = torch.zeros((self.br, d))
        L = torch.zeros((self.N))
        Lnew = torch.zeros((self.br))
        Lij = torch.zeros((self.br))
        Li = torch.zeros((self.br))

        if self.isRef:
            with record_function("STUDENT - FLASH ATTENTION"):
                out = mr.myFlashAttention(self.Q, self.K, self.V, Qi, Kj, Vj, Sij, Pij, PV, Oi, L, Li, Lij, Lnew, self.bc, self.br, self.B, self.H, self.N, self.d)
            return out
        with record_function("REFERENCE - FLASH ATTENTION"):
            #out = ms.myFlashAttention(self.Q, self.K, self.V, self.B, self.H, self.N, self.d, self.blockSize)
            out = ms.myFlashAttention(self.Q, self.K, self.V, Qi, Kj, Vj, Sij, Pij, PV, Oi, L, Li, Lij, Lnew, self.bc, self.br, self.B, self.H, self.N, self.d)
        return out

# generates dummy matrices for use in part0 
def createQKVSimple(N,d,B,H):
    Q = torch.empty(B,H,N,d)
    K = torch.empty(B,H,d,N)
    V = torch.empty(B,H,N,d)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for j in range(d):
                    Q[b][h][i][j] = 0.0002 * i + 0.0001 * j
                    K[b][h][j][i] = 0.0006 * i + 0.0003 * j
                    V[b][h][i][j] = 0.00015 * i + 0.0008 * j
    K=K.transpose(-2,-1)
    return Q,K,V


def test(Q,K,V):
    with profile(activities=[ProfilerActivity.CPU],
            profile_memory=True, record_shapes=True) as prof:

        start = time.time()
        #compute QK^T
        QK = Q @ K.transpose(-2,-1)
        #compute softmax of QK^T
        QKSoftmax = F.softmax(QK, dim=3)
        QKV = QKSoftmax @ V
        end = time.time()
        pytorch_time = end - start

        #compute QK^TV

        attentionModule = CustomAttention(Q,K,V)
        start = time.time()
        QKS1 = attentionModule.myFusedAttention()
        QKS2 = attentionModule.myFlashAttention()
        QKS2 = attentionModule.myUnfusedAttention()
        #QKS1 = ms.my_attention(Q, K, V)
        end = time.time()
        manual_time = end - start
        print("Pytorch Execution Time:", pytorch_time, "\n")
        print("Manual Execution Time: ", manual_time, "\n")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
        print(Q.shape)
        print(Q.stride())

def badSoftmax(Q, K, V):
    QK = Q @ K.transpose(-2,-1)
    #compute softmax of QK^T
    QKSoftmax = F.softmax(QK, dim=3)
    QKV = QKSoftmax @ V   
    return QKV

def testTemplate(customFunc, params, test_key):
    start = time.time()
    N, d, B, H = params
    #compute pytorch unfused softmax
    Q, K, V = createQKVSimple(N,d,B,H)
    QKV = badSoftmax(Q,K,V)
    end = time.time()
    pytorch_time = end - start

    with profile(activities=[ProfilerActivity.CPU],
            profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            #compute with Naive Unfused 
            start = time.time()
            QKS1 = customFunc()
            end = time.time()
            manual_time = end - start
    
    assert torch.allclose(QKV,QKS1, atol=1e-4), correctness_error_message
    print("manual attention == pytorch attention",torch.allclose(QKV,QKS1, atol=1e-4)) 
    #print("Pytorch Execution Time:", pytorch_time, "\n")
    print("Manual Execution Time: ", manual_time, "\n")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))    
    r = prof.key_averages()
    for rr in r:
        if rr.key == test_key:
            key, cpu_time, mem_usage = rr.key, rr.cpu_time, rr.cpu_memory_usage
            print (test_key+ " statistics")
            print("cpu time: ", str(cpu_time / 1000.0) + "ms")
            print("mem usage: ", mem_usage, "bytes")

def part0Test(N, d, B, H):
    print("Running part 0 test: Pytorch Matmul + Softmax")
    Q,K,V = createQKVSimple(N,d,B,H)
    with profile(activities=[ProfilerActivity.CPU],
            profile_memory=True, record_shapes=True) as prof:

        start = time.time()
        #compute pytorch unfused softmax
        QKV = badSoftmax(Q,K,V)
        end = time.time()
        pytorch_time = end - start

    print("Pytorch Execution Time:", pytorch_time, "\n")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


def part1Test(N, d, B, H):
    print("Running Part 1 Test: Naive Unfused Attention\n")
    Q,K,V = createQKVSimple(N,d,B,H)
    attentionModuleStudent = CustomAttention(Q,K,V, B, H, N, d)
    attentionModuleReference = CustomAttention(Q,K,V, B, H, N, d, True)
    params = (N, d, B, H)
    print("-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    testTemplate(attentionModuleStudent.myUnfusedAttention, params, "REFERENCE - NAIVE ATTENTION")
    time.sleep(3)
    print("-----RUNNING STUDENT IMPLEMENTATION-----\n")
    testTemplate(attentionModuleReference.myUnfusedAttention, params, "STUDENT - NAIVE ATTENTION")

def part2Test(N, d, B, H):
    print("Running Part 2 Test: Unfused Attention with Blocked Matmul\n")
    Q,K,V = createQKVSimple(N,d,B,H)
    attentionModuleStudent = CustomAttention(Q,K,V, B, H, N, d)
    attentionModuleReference = CustomAttention(Q,K,V, B, H, N, d, True)
    params = (N, d, B, H)
    print("-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    testTemplate(attentionModuleStudent.myUnfusedAttentionBlocked, params, "REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX")
    time.sleep(3)
    print("-----RUNNING STUDENT IMPLEMENTATION-----\n")
    testTemplate(attentionModuleReference.myUnfusedAttentionBlocked, params, "STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX")

def part3Test(N, d, B, H):
    print("Running Part 3 Test: Fused Attention\n")
    Q,K,V = createQKVSimple(N,d,B,H)
    attentionModuleStudent = CustomAttention(Q,K,V, B, H, N, d)
    attentionModuleReference = CustomAttention(Q,K,V, B, H, N, d, True)
    params = (N, d, B, H)
    print("-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    testTemplate(attentionModuleStudent.myFusedAttention, params, "REFERENCE - FUSED ATTENTION")
    time.sleep(3)
    print("-----RUNNING STUDENT IMPLEMENTATION-----\n")
    testTemplate(attentionModuleReference.myFusedAttention, params, "STUDENT - FUSED ATTENTION")

def part4Test(N, d, B, H, bc, br):
    print("Running Part 4 Test: Flash Attention\n")
    Q,K,V = createQKVSimple(N,d,B,H)
    attentionModuleStudent = CustomAttention(Q,K,V, B, H, N, d, False, bc, br)
    attentionModuleReference = CustomAttention(Q,K,V, B, H, N, d, True, bc, br)
    params = (N, d, B, H)
    print("-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    testTemplate(attentionModuleStudent.myFlashAttention, params, "REFERENCE - FLASH ATTENTION")
    time.sleep(3)
    print("-----RUNNING STUDENT IMPLEMENTATION-----\n")
    testTemplate(attentionModuleReference.myFlashAttention, params, "STUDENT - FLASH ATTENTION")

def accessTest(B, H, N, d):
    Q,_ ,_ = createQKVSimple(N,d,B,H)
    print("\nTensor Shape:", Q.size())
    print("\n4D Tensor Contents:\n", Q)
    b = random.randrange(B)
    h = random.randrange(H)
    i = random.randrange(N)
    j = random.randrange(d)
    print("\nIndexing Value When: x = " + str(b) + ", y = " + str(h) + ", z = " + str(i) + ", b = " + str(j))
    expected = round(Q[b][h][i][j].item(), 6)
    result = round(mr.fourDimRead(Q.flatten().tolist(), b, h, i, j, H, N, d), 6)
    print("Expected:", expected)
    print("Result:", result)
    assert abs(expected - result) < 1e-5
    
def main():

    d=32
    B=1
    H=4
    
    parser = argparse.ArgumentParser()
    parser.add_argument("testname", default="part0", help="name of test to run: part0, part1, part2, part3, part4, 4Daccess")
    parser.add_argument("-m", "--model", default="shakes128", help="name of model to use: shakes128, shakes1024, shakes2048, kayvon")
    parser.add_argument("--inference", action="store_true", default=False, help="run gpt inference")
    parser.add_argument("-bc",  default="256", help="Flash Attention Bc Size")
    parser.add_argument("-br", default="256", help="Flash Attention Br Size")
    parser.add_argument("-N", default="1024", help="Flash Attention Br Size")

    args = parser.parse_args()

    if args.model == "shakes128":
        N = 128
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes256":
        N = 256
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes1024":
        N = 1024
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes2048":
        N = 2048
        model_filename = "out-shakespeare-char2048Good"
    else:
        print("Unknown model name: %s" % args.model)
        return
    
    if args.inference == False:
        N = int(args.N)
        if args.testname == "part0":
            part0Test(N, d, B, H)
        elif args.testname == "part1":
            part1Test(N, d, B, H)
        elif args.testname == "part2":
            part2Test(N, d, B, H)
        elif args.testname == "part3":
            part3Test(N, d, B, H)
        elif args.testname == "part4":
            part4Test(N, d, B, H, int(args.bc), int(args.br))
        elif args.testname == "4Daccess":
            accessTest(1, 2, 4, 4)
        else:
            print("Unknown test name: %s" % args.testname)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)

        
if __name__ == "__main__":
    main()
