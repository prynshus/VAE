# Image Generation using Variational Autoencoders (VAE)

 This project uses variational autoencoders to generate images of MNIST dataset which are handwritten 0-9 digits. Our objective is to train the model and get 70% + accuracy for better results. This model helps to increase our idea of about neural networks which can be further used in GenAI (hot topic).

## What are variational autoencoders?

Variational Autoencoders (VAEs) are a type of generative model that learns to model the underlying distribution of data in an unsupervised manner. The term autoencoder consists of two parts:
1. Encoder:  Maps input data (e.g., images, text, etc.) to a lower-dimensional latent space.
2. Decoder:  Reconstructs the input data from the latent space representation.

The latent space is a compressed version of the data, where each point is a representation of the input. In VAEs, the latent space is stochastic, meaning the encoder outputs a probability distribution (typically Gaussian) rather than a single point.

## What is a convolutional layer?
A convolutional layer is a core building block of Convolutional Neural Networks (CNNs), which are a class of deep learning models primarily used for processing grid-like data, such as images or time-series data. 
It has three important features, kernel size, padding and stride. These features can be tuned to get better result.

We use this convolutional layers in our encoder.
## Architecture of our Model

In our case we have a 28 by 28 gray scale images (MNIST Dataset). Hence we use convolutional layer to reduce the size of these images into multiple channels.

![img.png](img.png)

Used two convolutional layers and two Linear layers with relu activation in the encoder.
Used two linear layer and two transposed convolutional layer with relu activation between hidden layers and sigmoid activation in the outermost layer of the decoder.

## Training the model
Used Binary cross entropy loss and adam optimizer for optimization.

## Result
Got 74.29% accuracy.

![epoch 3.png](result/epoch%203.png)

## Future improvement
1. Implementation of auto tuning for better results.
2. Use of reparametrisation trick in the autoencoders.
3. RGB datasets.

## References
https://youtu.be/JboZfxUjLSk?si=x4OzdELaXONc7S67

https://youtu.be/jDe5BAsT2-Y?si=Fv0ssp5We83JZUnj

https://youtu.be/xoAv6D05j7g?si=e1jmQwGrlRJRz4AS

https://youtu.be/z9hJzduHToc?si=FSQXd8qLTIdFBN1-

https://youtu.be/DPSXVJF5jIs?si=j8tMglLlLgcSQrPT