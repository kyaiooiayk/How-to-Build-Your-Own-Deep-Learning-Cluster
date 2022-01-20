# How to build your own Deep Learning cluster *(and what to know about GPUs!)*
This is an account of what I've read/done/followed to build my own deep learning cluster. Hopefully, it would be of some help for the others. It all started when I stumbled upon this [article](https://towardsdatascience.com/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252). What followed is a collection of questions and answers I asked myself while working on this project. Where possible and simply copy-and-shame the text as it was written by their original authors and their references is duly reported.

## Intended audience
Anyone who knows little about the topic and would like to get started somehow.

## What was my background?
I consider myself an aerospace engineer turned into a mix of software developer, CFD engineer, optimisation engineer, system engineer and data scientist. I had next-to-nothing experience on how hardware. If I did, you can also do it. The  only thing this tutorial is not going to provide you with is the capital to buy the component!

## My goal?
I'd like to:
- Learn a bit more about the hardware needed to run a DL network.
- Improve some of my coding skills.
- Have the ability to use the cluster when I want to with no restriction.
- Use it as a part of my portfolio as a Data Scientist (I know what you  are thinking!? This has nothing to do with Data Science! I agree with you, but I also true that being completely ingorant on the matter is equally bad! So, here I am.)
- Have a bit of fun doing it.
- Use to run some heavy-on-hardware Kaggle competitions (vision for instance).

## What am I not interested?
- I am not interested on buildng the latest moost powerful made-at-home cluster machine.
- I  am not interest into renting the machine to others.
- I do not want to use the freely available GPUs provided by Google and Kaggle for two reasons: I will be constrained in some manner and more importanlty my learning will be limited. At the end of the day, the idea of this project of mine was to learn a bit more about hardwares for DL.

## Some very high level questions
*(Read this if you are interested to quickly build a GPU cluster and less interested in understanding the inner working principles of GPUs)*
This a list of questions I had to answer myself before  spending my own money.
-  **Is this project going to make me bankrupt only by buying the component?** These references are difficult to compare as different prices mean different system, but this it is still good enough to get the right order of magniture. My conclusion is that with roughly a 3k GBP I should be able to get myself something. Here is a list with references of what I was able to gather from the internet:
   - [Ref #1](https://towardsdatascience.com/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252) | ~ $1,800 CAD | 2017
   - [Ref #2](https://towardsdatascience.com/building-your-own-deep-learning-box-47b918aea1eb) | ~ $1,600 USD | 2017
   - [Ref #3](https://www.oreilly.com/content/build-a-super-fast-deep-learning-machine-for-under-1000/) | ~ $1,000 USD | 2017

-  **Is this project going to make me bankrupt when running the machine?** How much is going to be my electricity bill?

-  **Can I run the two major deep learning frameworks (TensorFlow and PyTorch) without getting stuck into a "sorry, this hardware is currently not supported!"**

-  **How many CPUs and GPUs do I need?** From the references reported above there is seems to be consensus to use only one CPU but more than GPU if you can afford it. CPUs comes with some sort of hyperthreading, but people do not seem to bother toomuch about it. The real question is how many GPUs do I need? Each board comes with a certain number of physical GPUs. One of my goal is to use this in some Kaggle competitions. So to get an idea of many I'd ideally need I set myself into a quest of finding out how many GPUs kagglers used in the past competition. 
   -  [Ref #1]()
   -  [Ref #2]()
   -  [Ref #3]()

- **What are the most common GPU type used in homemade deep learning cluster?** If you have a large budget go for the TITAN X card, which is fantastic but costs at least $1,000, otherwise, if you have a finite budge, like most people, you can go for the NVIDIA GTX 900 series (Maxwell) or the NVIDIA GTX 1000 series (Pascal).
   - [Ref #1](https://towardsdatascience.com/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252) | CPU i5, 3.5GHz, Quad-Core | 1 GPU, GeForce GTX 1070 *8GB* SC Gaming ACX 3.0 Video Card
   - [Ref #2](https://towardsdatascience.com/building-your-own-deep-learning-box-47b918aea1eb) | CPU i7 4.2GHz Quad-Core | 1 GPU, Zotac GeForce GTX 1080 *8GB* AMP! Edition
   
-  **How many CPUs and GPUs can I afford?** It seems that the GPU are twice as expensive than a CPU and obviousy the factor will increase for some top-notch GPU. So for a budget of 2k spent only on GPU the best one can hope for is 4 GPUs.

- **How about the old K80?** According to this [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/#Is_upgrading_from_RTX_20_to_RTX_30_GPU_worth_it_Or_Should_I_wait_for_the_next_GPU) the K80 may offer you a good compromise between memory and speed. We see how memory is important to fit a model and how memory bandwith is also more important in terms of speed. So, can we traide some speed for more memory? If you have a limited budget then the K80 and the Quadro M6000 are alternatives you can consider. The Quadro M6000 has 24 GB of memory and the Tesla K80 has a 2-in-1 GPU with 2x 12 GB of memory. The more important question is whether the K80 have tesnor cores or not? According to these [Ref](https://www.microway.com/knowledge-center-articles/comparison-of-nvidia-geforce-gpus-and-nvidia-tesla-gpus/) they don't.

-  **Can I buy some second hand components?** There seems to be a market just for it. Some trusted website are:
   -  [Amazon second hand GPUs](https://www.amazon.co.uk/s?k=used+gpu&s=price-desc-rank&adgrpid=120731098785&gclid=CjwKCAiAv_KMBhAzEiwAs-rX1L2CB-E-AuMlTfSi6eDv8YxfinXuslQf7qOr7akmjG9_JI2COYLV1xoCOHsQAvD_BwE&hvadid=516377110988&hvdev=c&hvlocphy=1006567&hvnetw=g&hvqmt=e&hvrand=12132915950628804715&hvtargid=kwd-341756687644&hydadcr=17220_1714691&qid=1637668594&tag=googhydr-21&ref=sr_st_price-desc-rank).
- These [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/#What_is_NVLink_and_is_it_useful) provides a list how much you could end up spending for a second hand GPU (orer from new to low):
   -  RTX 2070 ($400) 
   -  RTX 2060 ($300)
   -  GTX 1070 Ti ($230)
   -  GTX 1070 ($220) 
   -  GTX 980 Ti (6GB $150) 
   -  GTX 1650 Super ($190)

-  **How dangerous are these machines in terms of fire safety?** What if my cooling system fails and I am not there to turn the machine off? 

-  **Can I get the same level of flexibility from freely available resource such as Google Colab or Kaggle?** Like it or not these resources are capped in way or the other. I am assuming that these caps will change over type, hence there is no point for me to report them here. My point here is that, if you're planning to make a heavy use of this machine with the final goal to learn about the hardware then go ahead and build your own. 

-  **How much support can I get from the internet?**

-  **What sort of internet connectivity do I need? Can I do it without?**

-  **How long can I keep the cluster for? Will I be forced to change the hardware fairly soon?** As per everythong in today technological world, nothing lasts forever. The question can be asked differently, will I be able to make good use of this for the next 5 years?

-  **Would I be better off buying more older GPUs or less but newer GPUs?** As long as we all understand the trade off between memory and speed for GPUs, I'd say that the main player in asnwer this question, would be your budget constraint.

-  **Where can I can I get access to some GPUs for free?** I know there are at least two options.
   -  [Google Colab](https://colab.research.google.com/signup) offers three plan: Colab free | Colab Pro | Colab Pro+. With the free memberhsip you get acces to K80 GPU, whereas pay subcribers get access to T4 and P100 GPUs.
   -  [Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage)  Kaggle provides free access to NVIDIA TESLA P100 GPUs. You can use up to a quota limit per week of GPU. The quota resets weekly and is 30 hours or sometimes higher depending on demand and resources 

- **Why not buying more CPUs?** For a very simple reasons. GPUs, if used properly, are simply the best for DL.

- **How much memory do I need?** I guess to answer this question we should know what do I need it for in the first place? My main goal was to use this cluster for some some Kaggle competitions, so I ran a search to understand what would be the best option for me. One of the things that is repeated over and over is the 11GB memory threshold. What I mean is that a GPU with more than 11GB of memory will allow you to have the ege on certain things. [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/) explains why. This is so because most previous models that are pretrained have pretty steep memory requirements, and these models were trained with at least RTX 2080 Ti GPUs that have 11 GB of memory. Below the 11GB memory threshold, the RTX 3070/3080 seems to offer the best cost-efficient setup. For prototyping, you want the largest memory, which is *still cheap*. With prototyping, I mean here prototyping in any area: Research, competitive Kaggle, hacking ideas/models for a startup, experimenting with research code. This [article](https://lambdalabs.com/blog/deep-learning-hardware-deep-dive-rtx-30xx/) compares the performance of the RTX 3070, 3080 and 3090. To recap here are some numbers:
    - Using pretrained transformers; training small transformer from scratch>= 11GB
    - Training large transformer or convolutional nets in research / production: >= 24 GB
    - Prototyping neural networks (either transformer or convolutional nets) >= 10 GB
    - Kaggle competitions >= 8 GB
    - Applying computer vision >= 10GB
    - Neural networks for video: 24 GB
    - Reinforcement learning =10GB + a strong deep learning desktop the largest Threadripper or EPYC CPU you can afford.
    
- **Can I use different GPU types?** Yes, you can! But you cannot parallelize efficiently across GPUs of different types. I could imagine a 3x RTX 3070 + 1 RTX 3090 could make sense for a prototyping-rollout split. On the other hand, parallelizing across 4x RTX 3070 GPUs would be very fast if you can make the model fit onto those GPUs. The only other reason why you want to do this that I can think of is if you’re going to use your old GPUs. This works just fine, but parallelization across those GPUs will be inefficient since the fastest GPU will wait for the slowest GPU to catch up to a synchronization point (usually gradient update). [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/#Is_upgrading_from_RTX_20_to_RTX_30_GPU_worth_it_Or_Should_I_wait_for_the_next_GPU)

- **What is NVLink, and is it useful?** NVLink is a high speed interconnect between GPUs. This is something used if you have a cluster with +128 GPUs; in other cases it probably not beneficial.

## CPUs vs GPUs
(*Read this if you are interested to understand the inner working principles of GPUs*)
- **How GPUs came about?** Developed by NVIDIA in 2007.

- **Only NVIDIA?** At the moment, there are no stable GPU computation platforms other than CUDA; this means that you must have an NVIDIA graphical card installed on your computer. CUDA was created by Nvidia. CUDA stands for Compute Unified Device Architecture. Nevertheless there are other project such as such as OpenCL that provide GPU computation for other GPU brands through initiatives such as [BLAS](https://github.com/clMathLibraries/clBLAS), but they are under heavy development and are not fully optimized for deep learning applications in Python. Another limitation of OpenCL is that only AMD is actively involved so that it will be beneficial to AMD GPUs. The hope is to have eventually a hardware-independent GPU application for machine learning. [Ref](https://www.packtpub.com/product/large-scale-machine-learning-with-python/9781785887215)

- **What is cache memory and why does it matter when understanding the difference between GPU and CPU?** Within the RAM, memory is not all equal. The computer will store a copy of part of RAM in what’s called the cache, a piece of memory that has much faster read/write time (around an order of magnitude faster). When you try to read a byte of RAM, the computer first looks in the cache. If that byte is stored there, it just reads the copy, only resorting to the actual RAM in the case of a “cache miss.” Similarly, if the program needs to modify a byte, it will first look for a cached copy of that byte and only modify that if it finds one. The computer will periodically write changes from the cache back to RAM in a batch process. Actually, there are usually several levels of cache, each one smaller and more rapid access than the one below it. The runtime of a program will often be dominated by how often the processor can find that data it’s looking for in the top levels of the cache. Together, the RAM, disk, and various cache levels form the “memory hierarchy,” where each layer is faster but smaller than the ones below it. Every time the computer needs a piece of data, it will look for it in the top level of the hierarchy, then the one below it, and so on, until the data is found. If a piece of data is located far down the hierarchy, then accessing it can be excruciatingly slow – partly because the access is inherently slow, but also because we just wasted time looking for the data higher up in the hierarchy. Thus, to conlude a good CPU-based code is usually designed to read information from the cache as much as possible. On GPU, most writable memory locations are not cached, so it can actually be faster to compute the same value twice, rather than compute it once and read it back from memory. [Ref#1](http://www.deeplearningbook.org/contents/applications.html), [Ref#2](https://www.amazon.co.uk/Data-Science-Handbook-Field-Cady/dp/1119092949)

- **How does GPUs work with CPUs?** How do they share the workload? GPU-accelerated computing is the employment of a graphics processing unit (GPU) along with a computer processing unit (CPU) in order to facilitate processing-intensive operations such as deep learning, analytics and engineering applications. GPU-accelerated computing functions by moving the compute-intensive sections of the applications to the GPU while remaining sections are allowed to execute in the CPU. [Ref](https://www.techopedia.com/definition/32876/gpu-accelerated-computing) 

- **Why GPUs have less cache but more memory bandwith?** Memory bandwidth is the rate at which data can be read from or stored into a semiconductor memory by a processor. On one side we have CPUs that have a deep parallel cache hierarchies which take up most of the space. On the other hand, you have GPUs that have loads of lighweight cores, that compared to CPUs, have much of their area devoted to arithmetic and less to memory and caches. This results in having much hegher memory bandwith. [Ref](http://www.cs.cornell.edu/courses/cs6787/2017fa/Lecture8.pdf)

   | GPUs                                    | CPUs |
   | --------------------------------------- | ---------------------------------------------- | 
   | Large No of cores, but slower than CPUs | Smaller No of cores, but much faster than GOPs |
   | High memory bandwith to control the cores | Lower memory bandwith |
   | Special purposes, meaning the do only task very very well | General purpouse |
   | Highly parallel processing | Sequential processing |

- **How does a cluster full of GPUs improve the situation when compared to a cluster full of CPUs?** This picture clarified my idea. [Ref](https://www.slideshare.net/ExtractConf/andrew-ng-chief-scientist-at-baidu)

![image](https://user-images.githubusercontent.com/89139139/148080348-b740fd05-e89c-4eb8-ac60-0a52f175a6ee.png)

- **How about the most outstanding issues GPUs suffer from?** They use a lot of electricity and thus produce a lot of heat which needs to be dissipated. This is not much of a problem as you’re training your neural network on a desktop workstation, a laptop computer, or a server rack. But many of the environments where deep learning models are deployed are not friendly to GPUs, such as self-driving cars, factories, robotics, and many smart-city settings where the hardware has to endure environmental factors such as heat, dust, humidity, motion, and electricity constraints. GPUs last around 2-5 years, which isn’t a major issue for gamers who usually replace their computers every few years. But in other domains, such as the automotive industry, where there’s expectation for higher durability! [Ref](https://bdtechtalks.com/2020/11/09/fpga-vs-gpu-deep-learning/)

- **Hold on a sec! How about TPUs?** Does it mean that I have to stop everything I am doing and wait for the TPUs to be made commercially available. No, I do not think so and I doubt Google will make them available to the general public anytime soon. A tensor processing unit (TPU) is an AI accelerator application-specific integrated circuit (ASIC) developed by Google specifically for neural network machine learning. TPUs are specifically optimized to perform fast, bulky matrix multiplication. TPUs are 15x to 30x faster than GPUs and CPUs respectively. TPU is a matrix multiplication engine — it does matrix multiplication and matrix operations, but not much else. You can freely access them [here](https://colab.research.google.com/notebooks/tpu.ipynb)

- **This is the best explaination I managed to find on the conceptual difference between GPUs and CPUs** As many have said GPUs are so fast because they are so efficient for matrix multiplication and convolution. Why? CPUs are latency optimized while GPUs are bandwidth optimized. You can visualize this as a CPU being a Ferrari and a GPU being a big truck. The task of both is to pick up packages from a random location A and to transport those packages to another random location B. The CPU (Ferrari) can fetch some memory (packages) in your RAM quickly while the GPU (big truck) is slower in doing that (much higher latency). However, the CPU (Ferrari) needs to go back and forth many times to do its job while the GPU can fetch much more memory at once. Currenlty (2020), the best CPUs have about 50GB/s while the best GPUs have 750GB/s memory bandwidth.
A big truck may be able to pick up a lot of packages with each tour, but the problem is that you are waiting a long time until the next set of packages arrives. Without solving this problem, GPUs would be very slow even for large amounts of data. So how is this solved? However, if you now use a fleet of either Ferraris and big trucks (thread parallelism), and you have a big job with many packages (large chunks of memory such as matrices) then you will wait for the first truck a bit, but after that you will have no waiting time at all — unloading the packages takes so much time that all the trucks will queue in unloading location B so that you always have direct access to your packages (memory). 
This effectively hides latency so that GPUs offer high bandwidth while hiding their latency under thread parallelism — so for large chunks of memory GPUs provide the best memory bandwidth while having almost no drawback due to latency via thread parallelism. 
This is the second reason why GPUs are faster than CPUs for deep learning. As a side note, you will also see why more threads do not make sense for CPUs: A fleet of Ferraris has no real benefit in any scenario. But there is 3rd advantages: GPUs have large and fast register and L1 memory which is easily programmable are the components which make GPUs so well suited for deep learning. Thus in order of importance: [Ref](https://www.quora.com/Why-are-GPUs-well-suited-to-deep-learning/answer/Tim-Dettmers-1) 
   - (1) High bandwidth main memory, 
   - (2) Hiding memory access latency under thread parallelism, and 
   - (3) Large and fast register and L1 memory which is easily programmable are the components which make GPUs so well suited for deep learning.

- **With or without tensore core?** GPUs come with tensor cores but also withouth them. Tensor Cores reduce the used cycles needed for calculating multiply and addition operations by ~16-fold; for a 32×32 matrix, from 128 cycles to 8 cycles. Tensor Cores reduce the reliance on repetitive shared memory access, thus saving additional cycles for memory access. Tensor Cores are so fast that computation is no longer a bottleneck. The only bottleneck is getting data to the Tensor Cores. Note that also TPUs  have tensor cores, the difference is only in how the memory is loaded and this seems to give TPUs the edge when it comes to matrix multiplications. [Ref](https://timdettmers.com/2018/10/17/tpus-vs-gpus-for-transformers-bert/)

- **What is a cycle and operation latency?** If a processor runs at 1GHz, it can do 10^9 cycles per second. Each cycle represents an opportunity for computation.  Most useful operations take longer than one cycle, so one new operation to start, it needs to wait for the number of cycles of time it takes for the previous operation to finish. This is also called the *operation latency*. [Ref](https://timdettmers.com/2018/10/17/tpus-vs-gpus-for-transformers-bert/)

- **What is a wrap?** The smallest unit of threads on a GPU is a pack of 32 threads which is called a *warp*. All memory operations on the GPU are optimized for warps. For example, loading from global memory happens at a granularity of 32*4 bytes, exactly 32 floats, exactly one float for each thread in a warp. [Ref](https://timdettmers.com/2018/10/17/tpus-vs-gpus-for-transformers-bert/)

- **Why would you use memory bandwith to compare GPUs?** Tensor Cores are very fast. So fast, in fact, that they are idle most of the time as they are waiting for memory to arrive from global memory. For example, during BERT Large training, which uses huge matrices — the larger, the better for Tensor Cores — we have a Tensor Core TFLOPS utilization of about 30%, meaning that 70% of the time, Tensor Cores are idle. This means that when comparing two GPUs with Tensor Cores, one of the single best indicators for each GPU’s performance is their memory bandwidth. [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)
   | GPU | Memory bandwith |
   | --- | -------------- | 
   | LA100 | 1,555 [GB/s] |
   | V100| 900 [GB/s] |

- **Why share memory, L1 cache size and registers are important?*** We know that the speed at which we transfer data is the real issue for GPUs. We also know that the faster the memory trasnfer is the more we can exploit the GPU real power. However, the faster the memory transfer is, the smaller it is. For this reason the transfer of data through the GPU's memory hierarchy is highly optimised. This hierarchy is as follows: (from slow to super fast) global memory -> local shared memory -> register. To get the speed-up order of magnitude consider that a matrix memory tile in shared memory is ~10-50x faster than the global GPU memory, whereas the tensor core’s register is ~200x faster than the global GPU memory. Remember thath a tile is a memory block in shared memory. [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)

## Good to know (practical solutions to common problems)
- **Can I limit the GPU power consumtion?** Yes, this can be done and the bigger question is how performance is impacted by this? [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/) suggests that a reduction of 50[W] for a max rating of 250[W] RTX 2081 Ti would reduce the performance by only 7%. The result was obtianed by running a 4x RTX 2080 Ti cluster on a 500 mini-batches BERT inference.
![image](https://user-images.githubusercontent.com/89139139/149536092-96b69ea5-98a5-4d1a-8df7-cdab58540935.png)
- **How can I fit +24GB models into 10GB memory?** A need made it worse since the onslaught of BERT-like models. Here is a list of solutions:
    - Training on 16-bit precision instea of the classical 32 as shown [here](https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c)
    - Gradient checkpointing (only store some of the activations and recompute them in the backward pass) as shown [here](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb)
    - GPU-to-CPU Memory Swapping (swap layers not needed to the CPU; swap them back in just-in-time for backprop) as shown [here](https://arxiv.org/abs/2002.05645v5)
    - Model Parallelism (each GPU holds a part of each layer; supported by fairseq) as shown [here](https://timdettmers.com/2014/11/09/model-parallelism-deep-learning/)
    - Pipeline parallelism (each GPU hols a couple of layers of the network)
    - ZeRO parallelism (each GPU holds partial layers)as shown [here](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
    - 3D parallelism (Model + pipeline + ZeRO) as shown [here](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)
    - CPU Optimizer state (store and update Adam/Momentum on the CPU while the next GPU forward pass is happening) 

## How I've made my final choice?
- The two main points are my (personal, your may be others!): constraint on budget and constrain on how much I was willing to pay in extra electricity. For me the electricity cap was more stringent than the actual lump sum to by the component. The reason is simple, if the electricity cost is too high then it will quickly sum up to a huge amount over time and the last thing I need is to by a Ferrari and not using it. You get the drift.

## Unbiased benchmarking
- Normalized GPU deep learning performance relative to an RTX 2080 Ti. [Ref](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)
![image](https://user-images.githubusercontent.com/89139139/149625891-75e45a9e-f927-452f-8454-66072dda5e32.png)



## References
- [Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)
