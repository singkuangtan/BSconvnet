# Newly added
Pytorch BSconv layer class codes to substitute your ordinary conv layer. BSnet neurons can be substituted into convolution, fully connected, attention and transformer layers neurons.

# BSconvnet
Boolean Structured Convolutional Deep Learning Network (BSconvnet)

# Main Takeaways

- Our model has only 17000+ parameters instead of 3.4 millions parameters in SmoothNet
- Use separable convolutional deep learning network, so it does not overfit
- Achieved state of the art accuracy on CIFAR10 dataset 
- Able to be trained on online using Google Colab with GPU
- No data augmentations, no regularization such as weight decay and dropout 

# How to Run

Download the jupyter notebook

Open using Google Colab
[Colab](https://colab.research.google.com/)
It can also be run using a jupyter notebook

Follow the steps in the notebook, run each block of codes starting from the top to the bottom

# Model

![Network design](https://github.com/singkuangtan/BSconvnet/blob/main/model1.png)
![Network design2](https://github.com/singkuangtan/BSconvnet/blob/main/model2.png)

# Experiment Results 

![Experiment results](https://github.com/singkuangtan/BSconvnet/blob/main/train1.png)
![Experiment results2](https://github.com/singkuangtan/BSconvnet/blob/main/train2.png)

![Training set embeddings](https://github.com/singkuangtan/BSconvnet/blob/main/train_embeddings.png)
![Test set embeddings](https://github.com/singkuangtan/BSconvnet/blob/main/test_embeddings.png)

Leaderboard of model accuracies on CIFAR10 dataset
[Leaderboard](https://paperswithcode.com/sota/image-classification-on-cifar-10)

![Leaderboard_pic](https://github.com/singkuangtan/BSconvnet/blob/main/table.png)

# Links
[BSnet paper link](https://vixra.org/abs/2212.0193)

[BSautonet paper link](https://vixra.org/abs/2212.0208)

[BSconvnet paper link](https://vixra.org/abs/2305.0166)

[BSnet GitHub](https://github.com/singkuangtan/BSnet)

[Discrete Markov Random Field Relaxation](https://vixra.org/abs/2112.0151)

[Slideshare](https://www.slideshare.net/SingKuangTan)

That's it. 
Have a Nice Day!!!
