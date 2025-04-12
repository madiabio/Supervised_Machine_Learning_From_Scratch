# Supervised Machine Learning Algorithms From Scratch
This repository contains two machine learning models I implemented from scratch in June 2024 for a university project on supervised machine learning. The first project used decision tree learning to
evaluate cars based upon metrics and the second used a 3 layer neural network to classify clothing items using image data. I implemented
the algorithms for both projects from scratch and then wrote tests to evaluated their performance based upon metrics I determined. This assignment was
highly influential in developing my interest in machine learning and data science. 

The directories for each model contain READMEs with much more information about the specific model.

I did not use GitHub for this project and retroactively created this repository to use as a platform to showcase my work.

# ID3 Decision Tree Learning
The first algorithm I implemented was the ID3 Decision Tree Learning algorithm. I used ID3 to evaluate cars based upon a labelled feature matrix of 1728 cars. This model classified cars into one of four classes, unacceptable, acceptable, good or very good, based upon their features. ID3 builds a decision tree by using entropy to determine the discriminatory power of each feature. Decisions are made by traversing the branches of the decision tree from the root node based upon how the feature vector’s value for the current node’s feature compares to the ground truth until the current node is a leaf/decision. At that point, a decision is returned. I obtained the learning curve of this model by iteratively training and then testing it on an increasingly large portion of the training set. 

# Neural Network
The second algorithm was a 3-layer neural network which I trained to classify fashion items based upon the fashion-mnist dataset from Zandolo research.  The model works by randomly initalising weights and biases and performing a forward pass on the network by updating the output nodes to the weighted sum of inputs with the sigmoid function applied as the activation function. Then, back propagation is performed on each layer to adjust the weights and biases by comparing the outputs of the network to the desired outputs of the test sample it has run on. Minibatching is used in this network to increase accuracy and works by essentially averaging out the weight and bias updates across some number of samples for each minibatch. Backpropagation is performed using all minibatches in the dataset. This neural network also uses the concept of epochs, where the network is trained on the training dataset multiple times. The highest number of epochs this network as trained on was 30. This means for each minibatch in the dataset, the network adjusted its weights and biases 30 times.
