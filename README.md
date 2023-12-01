# Assignment 4: NanoGPT149

**Due Monday Dec 4, 11:59pm PST**

**100 points total + 12 Points EC**

## Overview 

In this assignment, you will implement and optimize the key components of a transformer-based deep neural network that synthesizes Shakespeare text. While the DNN you will work with is a fairly small model, the basic components of the model are the same as those featured in large-language models (LLMs) that form the basis of technologies like ChatGPT today. Specifically, you will implement the attention layer of this model in C++, focusing on optimizations that improve arithmetic intensity, reduce memory footprint, and utilize multi-core and potentially SIMD parallelism on the CPU. Your implementation will then be used as part of a complete [NanoGPT](https://github.com/karpathy/nanoGPT) model that you can run to produce novel Shakespeare-like text.

Overall, this assignment will:

 * Give you experience with the low-level details of implementing DNN layers. In other words, the "guts" of vendor libraries like NVIDIA's CuDNN or Intel's One API.
   
 * Show the value of key locality-perserving optimizations like loop blocking and loop fusion.

## Environment Setup

We will provide you with SSH access to a cluster of shared machines for testing your code for this assignment. (We are not using AWS Lightsail like we did in Programming Assignment 3). You will directly log into these machines via ssh. Details on how to access the cluster will be provided in an Ed post.

To get started, clone the repo from github:

    git clone https://github.com/stanford-cs149/cs149gpt.git

Run the command below to run inference using a model trained by the CS149 staff. You will see some randomly generated Shakespeare text.

     python3 gpt149.py part0 --inference -m shakes128

Note that the first time you run the program, it will perform a compilation step that may take a few seconds, you'll see the text `Compiling code into a PyTorch module...`. <br><br>
After this is complete, you'll see some text that begins something like this:

    Running inference using dnn model shakes128
    number of parameters: 0.80M
    Loading meta from data/shakespeare_char/meta.pkl...

    BOTtaps along my lord.

    DUKE OF AUMERLE:
    The this is needs! Camillo, put I will make be strong.

    QUEEN MARGARET:
    My lord, yet t
    -------------------------------------------------------------
    CAMILLO:
    The shadows men sweet thy will burn comes.
    
    FLORIZEL:
    But of appear, good from thy heart
    As I be come of repeal of a w
    -------------------------------------------------------------

Sure, NanoGPT's output may not be literary excellence, but it is still pretty neat! What you see on screen is the output of the standard PyTorch implementation of NanoGPT. Feel free to change to larger sequence lengths by changing the `-m` parameter to larger models like `shakes256`, `shakes1024`, or `shakes2048`. You'll see the performance of NanoGPT token generation slow considerably with the bigger models.

### My Compilation Hangs
Some students have experienced issues when their compilation randomly starts hanging even though it was working before. When Python JIT compiles your code, it uses locks so multiple threads can compile it as once for efficiency. If you ever compiler your code and it hangs it means that for some reason Python thinks that the lock to your file is held. In order to fix this you can run:

     rm ~/.cache/torch_extensions/py310_cpu/custom_module/lock

## An Attention Module

The NanoGPT module you are executing in this assignment is a sequence-to-sequence model. The input is a sequence of words, such as the phrase *"The course of true love never did run smooth"*.  And the output of the model is a new sequence of words that is likely to follow the input, as determined by a model that have been trained on a large corpus of Shakespeare text. For example, given the prefix above, the output of the model might be *"whispered cs149 students whilst coding on assignments"*.  

The NanoGPT model uses a popular DNN module called a *transformer*, and a key component of a transformer module is a block called the *attention mechanism*. In this assignment your job is to implement the attention mechanism.  You will begin with a simple sequential implementation of attention, and then over the course of the assignment we'll take you through the process of adding optimizations like loop blocking, loop fusion, and basic parallelization.

In this section, we will describe the math of the attention mechanism (that you are supposed to compute). You may refer to [Slide 52 of Lecture 10](https://gfxcourses.stanford.edu/cs149/fall23/lecture/dnneval/slide_52) for a visual representation to follow along with. To students that seek more intuition for *why* an attention mechanism is what it is, we refer you to the many online tutorials about this popular DNN architecture such as:

  * [What is the intuition behind the attention mechanism?](https://ai.stackexchange.com/questions/21389/what-is-the-intuition-behind-the-attention-mechanism)
  * [Transformer Neural Networks: A Step-by-Step Breakdown](https://builtin.com/artificial-intelligence/transformer-neural-network)
  * [How Transformers Work](https://towardsdatascience.com/transformers-141e32e69591)

The attention mechanism takes as input three matrices `Q`, `K`, and `V`, referred to as "query", "key", and "value" vectors.  Each of these matrices are `Nxd` in size. `N` is the number of tokens (words) in the input sequence, so each row in these matrices is a length-`d` vector containing an embedding (a neural code) for one of the input words.  In other words `Q`, `K`, and `V` all contain different `d`-dimensional embeddings of the input tokens.

**Important Caveat:** To increase the efficiency and expressive power of the model, this attention module is typically run multiple times in parallel, due to there being multiple attention heads and multiple inputs in a batch. Understanding exactly why this is the case is not important, but just know that, in your implementation, these matrices will appear as **4D** tensors, where you will just be concerned with two of the four dimensions (corresponding to the $N\times d$ size.)

The first step of an attention module is to compute all pairs of interactions between the words.  This is done by multiplying the query matrix $Q$ against the key matrix $K$ to compute:

$$S = QK^T.$$

The next computation is a [softmax operation](https://machinelearningmastery.com/softmax-activation-function-with-python/) performed per-row of $S$.  The softmax produces normalized probabilities per row. 

For each row of the matrix, the softmax operation performs the following computation. Note that we give you the math for computing a softmax on a 1D vector $X$.  You'll need to perform this math for each row of the matrix $S$ above.

$$\text{softmax}(x) = \frac{\mathbf f(x)}{l(x)}$$

where

$$\mathbf f(x) = \begin{bmatrix}e^{x_1} & e^{x_2} &\cdots & e^{x_N} \end{bmatrix}\qquad \text{and} \qquad l(x) = \sum_{i=1}^N f(x)_i.$$

Note that the math above differs from the equation you saw in lecture in that `max(x)` is not subtracted from the numerators. The version of the math in lecture is what's used in practice for numerical stability, but in this assignment you can just use the math above. (It's a lot easier to implement FlashAttention later in the assignment if you use the math above.) For the ambitious, if you wish to use the version from lecture, feel free... you may see differences in "correctness checking".)

This yields a matrix of attention weights $P$, where

$$P = \texttt{softmax}(\texttt{each row of }S).$$

Finally, the attention weights are used to aggregate a set of learned **value** vectors, which are provided as a matrix $V$ of shape $N \times d$, to produce a final output $O$:

$$O = PV.$$

In summary, the attention layer consists of an expensive matrix multiply, followed by a softmax layer, followed by one more matrix multiply. These three components will be the bulk of your implementation -- read on for the details!

## Warm-Up: Accessing Tensors (3 Points)
Tensors are a data abstraction used in Pytorch. Although the name seems a bit intimidating, they are nothing more than multi-dimensional arrays. By abstracting these multi-dimensional arrays into a tensor datatype, the average PyTorch programmer no longer has to worry about how internals such as accessing a value or matrix multiplication work. Furthermore, Pytorch's tensors allow for easy GPU portability so that they can be run on specialized hardware, like Tensor cores. However, for this assignment we will be using CPU only, and instead of working with tensor datatypes we want you to work with a datatype you are all familiar with: C++ vectors.

A central key to understanding tensors is to know how to access them. This is why for the warm-up we want you to write accessors for a 4D tensor. For Parts 1-4, we have taken the liberty of providing you with a function called  `formatTensor` that transforms Tensors into C++ vectors. This provides you with a contiguous memory layout for a tensor's values, similar to how Pytorch stores tensor data. For Parts 1-4, we have also taken the liberty of transforming the output vector back into a tensor for you.

### Step 1: Understand a 2D Accessor
You should be relatively familiar with how to flatten a 2D array after Assignment 3. Your first job will be to understand how we can access an element of a multidimensional array, which is really just stored as a flattened 1D buffer in memory. We have given you example accessors for reading and writing to a 2D array. You will find them at the top of `module.cpp` under the names `twoDimRead` and `twoDimWrite.` The given 2D accessors will show you how we can access an arbitrary element `(i, j)` within this flattened array. **Note that the formula is as follows: For any given element (i, j) within an array, you may access it using [i * number_of_columns + j].**

### Step 2: Implement a 4D Accessor
In our LLM model, our arrays are 4D, so we will actually need a 4D accessor to access its elements! Now, extend the concepts behind accessing a 2D tensor so that you can access a 4D tensor. Your job for Step 1 is to implement the functions `fourDimRead` and `fourDimWrite` in the file `module.cpp`.

### Testing:
Run the following command to test your 4D accessor:

    python3 gpt149.py 4Daccess

When running the test, if you have implemented your accessors correctly, the expected and result values should be the same, resulting in an output like the one below. 

    Expected: 0.0006
    Result: 0.0006

### What to submit

* Implement `fourDimRead` and `fourDimWrite` in the file `module.cpp`.

* Next, answer the following question in your writeup:
  * Briefly describe how a 4D tensor/array is laid out in memory. Why do you think this convention was chosen and how does it leverage hardware?

## Part 1: A Simple (But Not So Efficient) Implementation of Attention (10 Points)

Now that you have your accessors, it's time to start working on your custom attention layer. As the first step in this assignment, you will be implementing a serial attention layer in C++ with no optimizations. In `myNaiveAttention`, we have provided you with two examples. The first demonstrates how to fill a 4D tensor with 0's and the second demonstrates how to fill a 2D tensor with 0's. Extend these concepts such that you are able to implement attention. You should:

    1) For each Batch:
    2) For each Head:
        a) Loop through Q and K and multiply Q with K^t, storing the result in QK^t. 
        QK^t is preallocated for you, and passed as an arg to myNaiveAttention. 
        (You should not have to allocate any pytorch tensors for any part of this assignment)
        
        Note that after indexing the batch and head, you will be left with 2D matrices 
        of shape (N, d) for Q and K. Also note that the dimensions of K are (N, d) and 
        the dimensions of the K^t that you want are (d, N). Rather than transposing K^t, how
        can you multiply Q and K in such an order that the result is QK^t? Think about how
        you can reorder your `for` loops from traditional matrix multiplication.
   
        b) After you have achieved QK^t -- which should have shape (N, N) -- you should loop 
        through each row. For each row, you should get the exponential of each row element,
        which you can get using the C++ inbuilt `exp` function. Now, divide each of these 
        resulting exponentials by the sum of all exponentials in its row and then store it back into QK^t. 
   
        c) Finally, you should matrix multiply QK^t with V and store the result into O. 
        Notice, much like Q and K, after you index the batch and head V and O will
        be of shape (N, d). Therefore, after you multiply QK^t (N, N) with V (N, d),
        you can simply store the resulting shape (N, d) back into O.

### Testing
Run the following test to check your program's correctness:

    python3 gpt149.py part1

While running the test, we show results of the pytorch profiler - this information is presented in a table which will show you detailed statistics on all function calls called in the test. The table that is dumped will look like the following:

    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  aten::empty         0.01%      23.000us         0.01%      23.000us       3.286us       5.00 Mb       5.00 Mb             7  
                  aten::zeros         0.14%     321.000us         0.18%     408.000us     102.000us       4.50 Mb           4 b             4  
    STUDENT - NAIVE ATTENTION        99.56%     229.600ms        99.97%     230.538ms     230.538ms       4.50 Mb      -1.00 Mb             1  
                  aten::clone         0.02%      37.000us         0.10%     231.000us     115.500us       1.00 Mb           0 b             2  
                aten::flatten         0.02%      48.000us         0.07%     153.000us      30.600us     512.00 Kb           0 b             5  
             aten::empty_like         0.00%       3.000us         0.00%       8.000us       8.000us     512.00 Kb           0 b             1  
          aten::empty_strided         0.01%      16.000us         0.01%      16.000us      16.000us     512.00 Kb     512.00 Kb             1  
              model_inference         0.02%      38.000us        99.98%     230.578ms     230.578ms     512.00 Kb      -4.00 Mb             1  
                  aten::zero_         0.02%      42.000us         0.15%     354.000us      88.500us           0 b           0 b             4  
                  aten::fill_         0.14%     312.000us         0.14%     312.000us     156.000us           0 b           0 b             2  
    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  

After the table is dumped, we also display two relevant statistics, cpu time (in milliseconds) and mem usage (in bytes). If you implemented your function correctly you should see those two values output like so:

    REFERENCE - NAIVE ATTENTION statistics
    cpu time:  230.724ms
    mem usage:  4718588 bytes
    
    STUDENT - NAIVE ATTENTION statistics
    cpu time:  232.561ms
    mem usage:  4718588 bytes

If your attention is not producing the correct output, you will see the following message:

    YOUR ATTENTION PRODUCED INCORRECT RESULTS
    
Note that you do not have to exactly match the reference `cpu time,` as long as you still produce a correct result. You should still be relatively close to the cpu time. We will provide you with a buffer of 15ms with respect to the reference cpu time. So, if you are <= 15ms behind the reference solution then that is fine and you will still get full credit. You are of course encouraged to beat the reference cpu time, and faster speeds will not be penalized.

Note that the memory usage value will not change even if you allocate more intermediate variables then we give you. This memory usage is only profiled from the variables passed in as arguments. For each Parts (1-4), we provide you with the minimum amount of variables to produce the correct result. **You can also assume that all the temporary intermediate tensors that we have passed in are initialized to contain zeros.** We do this because we want you to see how the memory usage goes down as operations get fused and there will be writeup questions based on these memory values. Adding any more high memory data structures will most likely only hurt your performance, but you are welcome to try adding additional variables in your `module.cpp` file and you will not be penalized.

You should be sure to test that your function works for different values of N, as this is how we will be grading you on correctness. We have provided a command line argument that works as so:

    python3 gpt149.py part1 -N <val>

If you have implemented your attention layer, you can also see the DNN use your attention layer to generate text, optionally changing the model to `shakes256`, `shakes1024`, or `shakes2048` if you wish to output more text:

    python3 gpt149.py part1 --inference -m shakes128

Note that you will not be autograded on inference, and this is purely for fun. Please also note that the models `shakes1024` and `shakes2048` will not work with the softmax we describe in this writeup due to overflow errors. If you wish to have them work, you must implement the "safe" softmax described in class. This is completely optional as we will always make sure to give you nice values when grading. Parts 1-3 all follow the same grading procedure listed here in this section.

### What to submit

* Implement `myNaiveAttention` in `module.cpp`.

## Part 2: Blocked Matrix Multiply and Unfused Softmax (20 Points)
Now that we have our baseline matrix multiply, let's see how we can optimize it. Currently, our matrix multiply behaves as follows:

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/current_matmul.png" width=40% height=40%>
</p>

Notice how poor the cache behavior of this operation is. For each element of C, we load in multiple cache lines from both A and B. However, something to keep in mind is that the size of these matrices are much bigger than our cache size. Therefore, by the time we want to process our next element of C, we will be reloading cache lines that have been evicted. But what if we reused these cache lines? The main reason that our code is inefficient is because we are processing a single element of C at a time, but what if we processed BLOCK elements at a time? In particular, what if we processed a cache line size of elements? 

Your job is to further extend your matrix multiply so that it employs blocking as discussed in [lecture](https://gfxcourses.stanford.edu/cs149/fall23/lecture/perfopt2/slide_43). You will decompose the large matrices into smaller cache-sized submatrices. Your multiply will then process the smaller submatrices before evicting them from the cache. The behavior should look like the following:

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/blocked_matmul.png" width=40% height=40%>
</p>

As a further example, let's say I have 3 NxN matrices and a cache line size of L. I would then break my 3 NxN matrices into (N/L)x(N/L) submatrices. How does this improve my cache utilization?

However, something to keep in mind is that we do not have perfectly square matrices. Our $Q$ and $K^{T}$ matrices are Nxd and dxN respectively. Keep this in mind as you try to break your matrices into blocks. Furthermore, your tile size will not always divide up N evenly, meaning you will have some "remainder" tiles that are not full of data. In this case, you do not want to iterate over the entire "remainder" tile, but only iterate over a "subtile", which has dimension `min(tile_size, N-tileIndex*tileSize)`.

Also, keep in mind that as before, the temporary memory you will need is already preallocated for you and passed to the function you need to implement (`myUnfusedAttentionBlocked`) - you shouldn't have to allocate anything yourself although you will not be penalized for doing so.

**Note that you have two opportunities for blocked matrix multiplication here: QK^t and PV. You should utilize blocked matrix multiply on both in order to achieve the reference speedup.**

### Testing:
Run the following test to check your program's correctness:

    python3 gpt149.py part2

A correct implementation should yield the following output:

    REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
    cpu time:  156.271ms
    mem usage:  4718588 bytes

    STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
    cpu time:  160.891ms
    mem usage:  4718588 bytes

An incorrect implementation will have the output:

    YOUR ATTENTION PRODUCED INCORRECT RESULTS

Just as in Part 1, we will autograde the correctness of your function's output and its CPU time. You have the same buffer of <=15ms of the reference solution. If your program is faster you will not be penalized.

You should be sure to test that your function works for different values of N, as this is how we will be grading you on correctness. We have provided a command line argument that works as so:

    python3 gpt149.py part2 -N <val>

You can see the DNN use your attention layer to generate text, optionally changing the model to `shakes256`, `shakes1024`, or `shakes2048` if you wish to output more text:

    python3 gpt149.py part2 --inference -m shakes128

Note that you will not be autograded on inference, and this is purely for fun. Please also note that the models `shakes1024` and `shakes2048` will not work with the softmax we describe in this writeup due to overflow errors. If you wish to have them work, you must implement the "safe" softmax described in class. This is completely optional as we will always make sure to give you nice values when grading.

### What to submit
* Implement `myUnfusedAttentionBlocked` in `module.cpp`.

* Then, answer the following questions in your writeup:
  * Share us some data about what tile sizes you tried when N=1024, and what the performance times were for each.  What was the optimal tile size for your matrix multiplications? Explain why you think this tile size worked best for your implementation. There really isn't a wrong answer here, we just want to see that you experimented and tried to form conclusions.
  * For a matrix multiply of $Q$ (Nxd) and $K^{T}$ (dxN), what is the ratio of DRAM accesses in Part 2 versus DRAM acceses in Part 1? (assume 4 byte float primitives, 64 byte cache lines, as well as N and d are very large).

## Part 3: Fused Attention (25 Points)
By now we've seen that multiplying $Q * K^{T}$ results in a massive NxN matrix. Doing the matrix multiplies and softmax in seperate functions requies that we write each row of our NxN matrix, and then do another pass over this NxN matrix in the subsequent softmax, and then do a third pass over the softmax'd matrix when multipling it by V. Not only is this bad for cache performance, but it is very bad for our program's memory footprint. 

Fortunately, we can resolve both issues by "fusing" the calculation, such that we only require one Nx1 temporary vector instead of an NxN temporary matrix.

You can do this by observing the following fact. Once we've calculated a single row of the $Q * K^t$ NxN matrix, we are actually ready to softmax that entire row, and we don't have to calculate the rest of the NxN matrix to do so.

Once that row is softmax'd, we can then immediately multiply the softmax'd row by V to fully compute the first row of our attention output (which is of reasonable size: Nxd). In other words, we can calculate just one row of $Q * K^{t}$, softmax it, then multiply that softmax's row by V. Doing this does not require creating the NxN matrix...it requires creating only one Nx1 size intermediate vector to hold the first row of $Q*K^{t}$ and then its softmax. We can then re-use this same Nx1 array to calculate the 2nd row of attention, and then the third, etc. This means that we never materialize the NxN matrix, which is great because that matrix is never used again later in the network anyways. 

### Parallelizing with OpenMP
As you may notice, now that we have fused our matrix multiplications and softmax, we made a significant portion of the computation embarrassingly parallel. For example, we are able to independently compute batches, heads, and rows of our output matrix. This is a perfect opportunity for multi-threading! This time we will be using OpenMP, so you don't have to implement your own threadpools. The OpenMP syntax is relatively simple. If you want to parallelize an section of code, it would look like the following:

    #pragma omp parallel for collapse()
    
    -- code is here --

You will find `#pragma omp parallel for collapse()` useful if you find loops directly nested on top of one another and want to parallelize them. For example, for a triple perfectly nested loop:

    #pragma omp parallel for collapse(3)

    for ()

        for()
    
            for()

Note: You'd usually want to be careful when writing to a single Nx1 temporary row when using OpenMP, as this is a race condition. To work around this, we give you a skeleton of the first three loops (you will need more loops) in which each OpenMP thread gets assigned its own copy of the Nx1 temporary array, in a way that avoids race conditions. This local copy of the array is a slice/subset of the temporary memory we allocate for you, and pass into the function (`myFusedAttention`) as an argument. Keep in mind that any variables declared inside the loop(s) you are trying to parallelize will be private to each thread.

### Testing:
Run the following test to check your program's correctness:

    python3 gpt149.py part3

A correct implementation should yield the following output:

    REFERENCE - FUSED ATTENTION statistics
    cpu time:  32.361ms
    mem usage:  557052 bytes

    STUDENT - FUSED ATTENTION statistics
    cpu time:  33.209ms
    mem usage:  557052 bytes

An incorrect implementation will have the output:

    YOUR ATTENTION PRODUCED INCORRECT RESULTS

Just as in Parts 1 & 2, we will autograde the correctness of your function's output and its CPU time. You have the same buffer of <=15ms of the reference solution. If your program is faster you will not be penalized.

You should be sure to test that your function works for different values of N, as this is how we will be grading you on correctness. We have provided a command line argument that works as so:

    python3 gpt149.py part3 -N <val>

Now, you can see the DNN use your attention layer to generate text, optionally changing the model to `shakes256`, `shakes1024`, or `shakes2048` if you wish to output more text:

    python3 gpt149.py part3 --inference -m shakes128

Note that you will not be autograded on inference, and this is purely for fun. Please also note that the models `shakes1024` and `shakes2048` will not work with the softmax we describe in this writeup due to overflow errors. If you wish to have them work, you must implement the "safe" softmax described in class. This is completely optional as we will always make sure to give you nice values when grading.

### What to submit
* Implement `myFusedAttention` in `module.cpp`.

* Then, answer the following question in your writeup:
  * Why do we use a drastically smaller amount of memory in Part 3 when compared to Parts 1 & 2?
  * Comment out your `#pragma omp ...` statement, what happens to your cpu time? Record the cpu time in your writeup. Why does fused attention make it easier for us utilize multithreading to a much fuller extent when compared to Part 1?

## Part 4 : Putting it all Together - Flash Attention (35 Points)
### Why Are Matrix Multiply and Softmax Hard to Fuse as Blocks?
The attention formula is very awkward to fuse for a couple reasons. Notice how the formula consists of a matrix multiply, followed by a row-wise calculation from softmax, and concluded with another matrix multiplication. The true thing that makes it difficult from fusing these three operations as blocks is the fact that softmax has to operate on the entire row. So, if we want to bypass this dependency we really have to think outside the box. That is where Flash Attention comes in.

### Breaking Softmax into Blocks
Let's say that we have a BLOCKSIZE vector, we will denote it as $x \in \mathbb{R}^{B}$.The softmax of $x$ can be formulated as:

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/Softmax_decomp1.png" width=55% height=55%>
</p>

It follows that if we have two BLOCKSIZE vectors, denoted as $x \in \mathbb{R}^{B}$ and $y \in \mathbb{R}^{B}$, then we can decompose $softmax([x\ y]$ as:

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/Softmax_decomp2.png" width=55% height=55%>
</p>


### Implement Flash Attention
Your job is to break softmax into blocks so it can be fused with your blocked matrix multiply. Therefore, for each block, you will multiply $Q$ (BLOCKROWSIZE x d) with $K^{t}$ (d x BLOCKCOLUMNSIZE) to get $QK^t$ (BLOCKROWSIZE x BLOCKCOLUMNSIZE). Then, you will calculate $\texttt{softmax}(QK^t)$ (BLOCKROWSIZE x BLOCKCOLUMNSIZE) and multiply this with $V$ (BLOCKCOLUMNSIZE x d) to get $O$ (BLOCKROWSIZE x d). Remember, this is an accumulative process just like blocked matrix multiply!

By doing this we can significantly decrease the memory footprint. Rather than having a memory footprint of $O(N^{2})$, we will be able to reduce this to a linear scaling footprint of $O(N)$.

### Flash Attention Pseudocode

The flash attention algorithm shown below, imports blocks of the matrices $Q$, $K$, and $V$ into smaller physical tiles. It then computes a local softmax in each tile, and then writes this result tile back to the full output matrix $O$. For $Q$, for example, each tile's size is (Br x d), and the tile size for $K$ is (Bc x d). Calculating $Br$ and $Bc$, as shown in the pseudocode below, requires knowing the size $M$ of your SRAM/cache, which in this case is $M=131072$ floats. For the purposes of this programming assignment, your program should be able to handle any $Br/Bc$ we give it.

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/FlashAttentionPseudo.png" width=65% height=65%>
</p>

### Testing:
Run the following test to check your program's correctness:

    python3 gpt149.py part4

**Make sure to test your implementation on different block sizes.** When running this test, the default values of $N$ and $d$ are $1024$ and $32$ respectively. Make sure that your program is able to handle any block size, whether your block size evenly divides into these values of $N/d$ or not. We have given you commandline flags to change the $Br$ and $Bc$ parameters of the attention algorithm. You can do this with the flags `-br <value>` and `-bc <value>`. The default values for each are $256$. For example, if I wanted to change $Br$ to $128$ and $Bc$ to $512$ I would run:

    python3 gpt149.py part4 -br 128 -bc 512

A correct implementation should yield the following output:

    REFERENCE - FLASH ATTENTION statistics
    cpu time:  435.709ms
    mem usage:  524284 bytes

    STUDENT - FLASH ATTENTION statistics
    cpu time:  435.937ms
    mem usage:  524284 bytes

An incorrect implementation will have the output:

    YOUR ATTENTION PRODUCED INCORRECT RESULTS

Notice that the cpu speed is actually lower than Part 3. Why is this the case? You will answer this question in your writeup below.

You should be sure to test that your function works for different values of N, as this is how we will be grading you on correctness. You should test different Ns as well as different block_sizes. Please note that the reference solution runs first, so if the reference solution fails then you do not have to worry about that combination of N/Br/Bc. To change the values of N, Br, and Bc:

    python3 gpt149.py part4 -N <val> -br <val> -bc <val>

**For this problem only**, you will be graded solely on correctness and not performance. The grading consists on an automated check that your algorithm produced the correct output and a manual check that you followed the pseudocode from above. If you ran the command `python3 gpt149.py part4` and you saw the output above that is associated with a correct implementation and DID NOT see: `YOUR ATTENTION PRODUCED INCORRECT RESULTS`, then you passed the autograded portion. For the correctness check, we also reserve the right to change the values of N, Br, and Bc. If you followed the pseudocode from the image above, then you will pass the manual check.

Now, you can see the DNN use your attention layer to generate text, optionally changing the model to `shakes256`, `shakes1024`, or `shakes2048` if you wish to output more text:

    python3 gpt149.py part4 --inference -m shakes128

Note that you will not be autograded on inference, and this is purely for fun. Please also note that the models `shakes1024` and `shakes2048` will not work with the softmax we describe in this writeup due to overflow errors. If you wish to have them work, you must implement the "safe" softmax described in class. This is completely optional as we will always make sure to give you nice values when grading.

### What to submit
* Implement `myFlashAttention` in `module.cpp`. 

* Then, answer the following question in your writeup:
  * How does the memory usage of Part 4 compare to that of the previous parts? Why is this the case?
  * Notice that the performance of Part 4 is slower than that of the previous parts. Have we fully optimized Part 4? What other performance improvements can be done? Please list them and describe why they would increase performance.

## Extra Credit: Optimize Further (12 Total Points - 3 Points Per Part)

### Vectorize with ISPC Intrinsics
You may notice that there are many looped-based nondivergent floating point operations. This is a great place to use vector intrinsics! We have provided ISPC support for you to write you own vectorized functions for things such as matrix multiplication and row sum. The repo contains a file titled `module.ispc`. Feel free to write your own ISPC functions in here, and compile them with the command:

     ispc -O3 --target=avx2-i32x8 --arch=x86-64 --pic module.ispc -h module_ispc.h -o module_ispc.o 
     
To enable them in your `module.cpp` file, all you need to simply uncomment the following two lines at the top of the file:

    #include "module_ispc.h"
    using namespace ispc;

### Write-Up Question
* Please record your speedups with vectorization and your implementations in `writeup.pdf.`

## Point BreakDown: (100 Total Points + 12 Possible Extra Credit)
* Implement `fourDimRead`: 1.5 Points
* Implement `fourDimWrite`: 1.5 Points
* Implement `myNaiveAttention`: 10 Points
* Implement `myUnfusedAttentionBlocked`: 20 Points
* Implement `myFusedAttention`: 25 Points
* Implement `myFlashAttention`: 35 Points
* Answer Writeup Questions: 7 Points
  * 1 Warm-Up Question
  * 2 Part 2 Questions
  * 2 Part 3 Questions
  * 2 Part 4 Questions
* Extra Credit: Vectorize Parts 1-4: 3 Points Per Part

## Hand-in Instructions
Please submit your work using [Gradescope](https://www.gradescope.com/). If you are working with a partner please remember to tag your partner on gradescope.

Please submit your writeup questions in a file `writeup.pdf`. REMEMBER to map the pages to questions on gradescope. If you did the extra credit, please state so at the end of your writeup as we will manually run these. In addition, please record the performance numbers we should expect for each part that you sped up using vectorization.

* Please submit the following files to Assignment 4 (Code):
  * module.cpp
  * module.ispc (if you attempted the extra credit)
    
* Please submit your writeup in a file called `writeup.pdf` to Assignment 4 (Write-Up).
