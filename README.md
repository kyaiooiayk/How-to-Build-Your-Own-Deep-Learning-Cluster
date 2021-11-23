# How to build your own Deep Learning cluster
This is an account of what I've read/done/followed to build my own deep learning cluster. Hopefully, it would be of some help for the others. It all started when I stumbled upon this [article](https://towardsdatascience.com/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252).

## Intended adicience
Anyone who known little about the topic and we'd like to get started somehow.

## What was my background?
I consider myself an aerospace engineer turned into a mix of software developer, CFD engineer, optimisation engineer, system engineer and data scientist. I had next-to-nothing experience on how hardware. If I did, you can also do it. The  only thing this tutorial is not going to provide you with is the capital to buy the component.

## My goal?
I'd like to:
- Learn a bit more about the hardware needed to run a DL network.
- Improve some of my coding skills.
- Have the ability to use the cluster when I want to with no restriction.
- Use it as a part of my portfolio as a Data Scientist (I know what you  are thinking!? This has nothing to do with Data Science! I agree with you, but I also true that being completely ingorant on the matter is equally bad! So, here I am.)
- Have a bit of fun doing it.
- Use to run some heavy-on-hardware Kaggle competitions (vision for instance).

## What I do not want?
- I am not interested on buildng the latest moost powerful made-at-home cluster machine.
- I  am not interest into renting the machine to others.
- I do not want to use the freely available GPUs provided by Google and Kaggle for two reasons: I will be constrained in some manner and more importanlty my learning will be limited. At the end of the day, the idea of this project of mine was to learn a bit more about hardwares for DL.

## Some very high level questions
This a list of questions I had to answer myself before  spending my own money.
-  **Is this project going to make me bankrupt only by buying the component?** These references are difficult to compare as different prices mean different system, but this it is still good enough to get the right order of magniture. My conclusion is that with roughly a 3k GBP I should be able to get myself something. Here is a list with references of what I was able to gather from the internet:
   - [Ref #1](https://towardsdatascience.com/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252) | ~ $1,800 CAD | 2017
   - [Ref #2](https://towardsdatascience.com/building-your-own-deep-learning-box-47b918aea1eb) | ~ $1,600 USD | 2017
   - [Ref #3](https://www.oreilly.com/content/build-a-super-fast-deep-learning-machine-for-under-1000/) | ~ $1,000 USA | 2017
-  Is this project going to make me bankrupt when running the machine? How much is going to be my electricity bill?
-  Can I run the two major deep learning frameworks (TensorFlow and PyTorch) without getting stuck into a "sorry, this hardware is currently not supported!"
-  **How many CPUs and GPUs do I need?** From the references reported above there is seems to be consensus to use only one CPU board. This generally come with some sort of hyperthreading, but people do not seem to bother toomuch about it. The real question is how many GPUs do I need? Each board comes with a certain number of physical GPUs. Once of my goalis to use this in the Kaggle competition. So to get an idea of many I'd ideally need I set myself into a quest of finding out how many GPUs kagglers used in the past competition. 
   - [Ref#1](https://towardsdatascience.com/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252) | CPU i5, 3.5GHz, Quad-Core | 1 GPU, GeForce GTX 1070 *8GB* SC Gaming ACX 3.0 Video Card
   - [Ref#2](https://towardsdatascience.com/building-your-own-deep-learning-box-47b918aea1eb) | CPU i7 4.2GHz Quad-Core | 1 GPU, Zotac GeForce GTX 1080 *8GB* AMP! Edition
-  **How many CPUs and GPUs can I afford?** It seems that the GPU are twice as expensive than a CPU and obviousy the factor will increase for some top-notch GPU. So for a budget of 2k spent only on GPU the best one can hope for is 4 GPUs.
-  **Can I buy some second hand components?** There seems to be a market just for it. Some trusted website are:
   -  [Amazon second hand GPUs](https://www.amazon.co.uk/s?k=used+gpu&s=price-desc-rank&adgrpid=120731098785&gclid=CjwKCAiAv_KMBhAzEiwAs-rX1L2CB-E-AuMlTfSi6eDv8YxfinXuslQf7qOr7akmjG9_JI2COYLV1xoCOHsQAvD_BwE&hvadid=516377110988&hvdev=c&hvlocphy=1006567&hvnetw=g&hvqmt=e&hvrand=12132915950628804715&hvtargid=kwd-341756687644&hydadcr=17220_1714691&qid=1637668594&tag=googhydr-21&ref=sr_st_price-desc-rank).
-  **How dangerous are these machines in terms of fire safety?** What if my cooling system fails and I am not there to turn the machine off?
-  
-  **Can I get the same level of flexibility from freely available resource such as Google Colab or Kaggle?** Like it or not these resources are capped in way or the other. I am assuming that these caps will change over type, hence there is no point for me to report them here. My point here is that, if you planning to make a heavy use of this machine with the final goal to learn about the hardware then go ahead and build your own. For seak of completion these a list of offereing from the major polayers:
   -  [Google Colab](https://colab.research.google.com/signup) offers three plan: Colab free | Colab Pro | Colab Pro+. With the free memberhsip you get acces to K80 GPU, whereas pay subcribers get access to T4 and P100 GPUs.
   -  [Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage)  Kaggle provides free access to NVIDIA TESLA P100 GPUs. You can use up to a quota limit per week of GPU. The quota resets weekly and is 30 hours or sometimes higher depending on demand and resources 
-  **How much support can I get from the internet?**
-  **hat sort of internet connectivity do I need? Can I do it without?**
-  **If I invest my money how long can I keep this cluster for? Will I be forced to change the hardware fairly soon?** As per everythong in today technolofical world nothing last forever. The question can be asked differently, will I be able to make good use of this for the next 5 years?
-  **Would I be better off buying more older GPUs or less but newer GPUs?**
-  **Where can I can I get access to some GPUs for free?** I know there are at least two options. Option No.1 is offered by [Google Colab](https://colab.research.google.com/?utm_source=scs-index). Option No.2 is offered by [Kaggle](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu).
-  **Why not buying more CPUs?** For a very simple reasons. GPUs, if used properly, are simply the best for DL.*
