# How to build your own Deep Learning cluster
This is an account of what I've read/done/followed to build my own deep learning cluster. Hopefully, it would be of some help for the others. It all started when I stumbled upon this [article](https://towardsdatascience.com/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252).

## What was my background?
I consider myself an engineer turned into a mix of software developer, CFD engineer, optimisation engineer, system engineer and data scientist. I had next-to-nothing experience on how hardware. If I did, you can also do it. The  only thing this tutorial is not going to provide you with is the capital to buy the component.

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
-  Is this project going to make me bankrupt when running the machine? How much is going to be my electricity bill?
-  Can I run the two major deep learning frameworks (TensorFlow and PyTorch) without getting stuck into a "sorry, this hardware is currently not supported!"
-  How many CPUs and GPUs do I need?
-  How many CPUs and GPUs can I afford?
-  Can I buy some second hand components?
-  How dangerous are these machines in terms of fire safety? What if my cooling system fails and I am not there to turn the machine off?
-  Can I get the same level of lfexibility from freely avaiilable resource such as Google Colab?
-  How much support can I get from the internet?
-  What sort of internet connectivity do I need? Can I do it without?
-  If I invest my money how long can I keep this cluster for? Will I be *forced* to change the hardware?
-  Would I be better off buying more older GPUs or less but newer GPUs?
-  **Where can I can I get access to some GPUs for free?** I know there are at least two options. Option No.1 is offered by [Google Colab](https://colab.research.google.com/?utm_source=scs-index). Option No.2 is offered by [Kaggle](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu).
-  **Why not buying more CPUs?** For a very simple reasons. GPUs, if used properly, are simply the best for DL.*
