# Captcha Recognition

### How to run the project?

**Important: I'm not sure that the code is going to run correctly in another systems other than my laptop because I have some problems related to inconsistencies of Tensorflow and Mac OS systems with ARM(M1) architecture. So please accept my apology if it didn't go well. I actually included my colab notebook in this folder because of this issue and I have good explanations about my work there.**

Before running the project, please make sure that the modules such as Tensorflow, numpy and matplotlib are installed on your virtual environment.

To run the project:

```
python3 solution.py
```

Actually this is the first model I implemented for this project and it got better results than others because I couldn't tune them to work better than this model. In the next section I will explain what I've done.

### My Colab notebook?

You can access my notebook using this [link](https://colab.research.google.com/drive/12KB8nkRddARifAwXb_ttDbTxopj8D0Xu?usp=sharing) or using `solution.ipynb` file.

### What I've done in this project?

Before startring the project I knew that we need a CNN model or a combination of CNN and RNN to solve this problem. So, I started to search a bit about captcha recognition networks and I found a variety of networks such as:

- A simple CNN network 

- ResNet

- DenseNet especially DFCR

- A simple CNN network + RNN using CTC loss function

So, I started to implement this architectures and I implemented the first 3 of them. Honestly, I had no time to implement the fourth one :(

#### 1.A simple CNN

When I was searching for some network to solve captchas, I find this [link](https://medium.com/@manvi./captcha-recognition-using-convolutional-neural-network-d191ef91330e) which explains a simple CNN network containing 3 convolutions and 2 fully connected layers which are branched into 5 different layers to detect captchas.

Having inspired by this model, I added another convolution to the model and also dropout layers after each convolutions and I reached a result nearly as same as that model. **(More than 90% accuracy for each digit)**

However, I knew that we can do even better than this. So, I decided to implement other models as well.

#### 2.ResNet50

I used Keras functional API to implement ResNet50 but unfortunately the result wasn’t good enough. Actually the reason is that I couldn't tune the network well and I didn't find a good combination of hypermarameters. But it was a good experience for me because I haven't had implemented ResNet from scratch before.

#### 3.DFCR (DenseNet)

As you might know, DenseNet is one the most powerfull CNNs and I found this when I was researching for this project and I wasn't familiar with it before. I found an [article](https://www.aimspress.com/fileOther/PDF/MBE/mbe-16-05-292.pdf) about some improvements on DenseNet to solve captchas more efficiently. So I went to read the DenseNet article first and then I read its improvment article and I was so hyped to implement this network. 

After implementing the network, I had few time to tune the network but unfortunately (again) I couldn't tune it and the result was not what I expected. But I'm pretty sure that this network can work much better than those other two networks. It was so exciting for me that how with such a creative idea we can have a network with far fewer parameters than ResNet but a more powerful one.

### How is the project organized?

I managed to organize the project in some different python files:

- `solution.py`: The main module of the project

- `layers.py`: This is where I've implemeted my new layers using keras functional API and with subclassing the `keras.layers.Layer` class.

- `models.py`: This is where I've implemeted 3 classes to build those 3 model I mentioned before.

- `schedulers.py`: I've implemented two learning rate scheduler (AdaptiveScheduler and StepDecay) in this module.

- `data_utils.py`:  The module in which I've implemented some functions to preprocess data, create datasets, batchify, ...

- `utils.py`: In this module we have some functions to plot the results of the training and also a function to set the seeds to make reproducable results.
