# Advanced Machine Learning Notes

## Linear Perceptron (Heaviside Function 𝜎(𝑥))
- A simple neural network model that uses a step function for activation.

## Multi-Layer Perceptron (MLP)
- **Introduction**
    - A Multi-Layer Perceptron (MLP) consists of multiple layers of neurons that help solve complex problems by introducing non-linearity through activation functions.
- **Activation Functions**
    - Sigmoid: Output range [0,1]
    - Hyperbolic Tangent (tanh): Output range [-1,1]
    - ReLU (Rectified Linear Unit): max(0, x)
- **Multi-Layer Perceptron Algorithm**
    - Forward propagation
    - Backward propagation
    - Stochastic Gradient Descent (SGD): An optimization method that updates weights after each sample to speed up learning.
- **Training Components**
    - The Loss: Measures prediction error
    - One Epoch: One full pass through the dataset
    - The Gradient: The derivative of the loss function used to update weights
    - Descent Rule: Algorithm to adjust weights based on gradients

## Keras and TensorFlow for implementing MLP

## Optimizers
- Determines how weights are updated. Common optimizers include:
    - Stochastic Gradient Descent (SGD): Uses a learning rate to define the step size
    - Adam Optimizer: Uses previous gradients to improve updates
    - Momentum: Helps maintain gradient direction
    - Batch Gradient Descent: Updates weights after processing a batch of data

## XOR Problem and Non-Linearity
- XOR is non-linear and cannot be solved using a simple perceptron. Activation functions help introduce non-linearity to solve such problems.

## Output Layer Activation Functions
- **Binary Classification**: One node with sigmoid (output: probability between 0 and 1) example: 0.7 
- **Multiclass Classification**: Softmax activation (mutually exclusive probabilities) example: [0.1, 0.7, 0.2] sum = 1
- **Multilabel Classification**: Sigmoid activation (allows multiple labels per sample) example: [0,1,1,0] sum can be more than 1

## Most Used Activation Functions
- ReLU: Avoids vanishing gradient issues
- ELU: Improved version of ReLU
- Sigmoid: Used for probabilistic outputs

## Neural Network Complexity
- More hidden layers → Higher complexity
- More neurons per layer → Greater ability to solve complex problems

## Learning Rate (n)
- Determines the step size for updating weights. A key hyperparameter in optimization.

## Loss Functions
- Binary Cross-Entropy: For binary classification
- Categorical Cross-Entropy: For multi-class classification (with one-hot encoding)

## Batch Processing
- **Batch**: A subset of the training data used for updating weights
- **Mini-batch Gradient Descent**: A mix of SGD and batch gradient descent

## Batch Normalization and Covariate Shift
- **Batch Normalization**: Normalizes input data (mean 0, variance 1)
- **Covariate Shift**: When the input data distribution changes over time
- **Link**: Batch normalization reduces covariate shift by stabilizing input distributions
- **Example**: In image recognition, if a cat appears on different backgrounds, batch normalization helps adapt to new distributions, reducing the need for relearning.

## Regularization Techniques
- **Dropout**: Randomly disables some neurons to prevent overfitting
- **Regularization**: Adds a penalty to the loss function to reduce overfitting

## Flattening
- Transforms multi-dimensional input data into a 1D array before passing it to dense layers.

## Convolutional Neural Networks (CNNs)
- CNNs are specialized neural networks for image processing tasks.
- **Kernel (Filter)**: Extracts features from input data
- **Patch**: A small region of the input data that the kernel processes
- **Kernel Size**: Determines filter dimensions (hyperparameter)
- **Hyperparameters**: Kernel size, depth, max pooling
- **Stride**: Defines how far the kernel moves across the input data (le pas)
- **Output Size Formula**: (input size - kernel size) / stride + 1
- **Padding**: Adds zeros around the input to prevent loss of information when applying convolutions.
- **TensorFlow**: Can automatically determine output size and handle padding for optimal feature extraction.

- The kernel weights are learned through backpropagation. The network adjusts the weights to minimize the loss function, which measures the difference between predicted and actual values. The kernel weights are updated based on the gradients of the loss function with respect to the weights. This process is repeated iteratively until the model converges to a solution.

- **Architecture Example**: input image -> convolutional layer -> pooling layer -> fully connected layer -> output layer
- **Flattening**: The process of converting the pooled feature map to a single column that is passed to the fully connected layer.

- To reduce training time, pooling layers are used to reduce the spatial dimensions of the feature map.

- **Pooling Layers**: Reduce the spatial dimensions of the feature map by downsampling the data.
    - **Max Pooling**: Selects the maximum value from the pool
    - **Average Pooling**: Calculates the average value

- Problems are often non-linear, so activation functions are used to introduce non-linearity into the model.
- ReLU is the most commonly used activation function in CNNs due to its simplicity and efficiency.

- **Training a CNN with Multiple Convolutional Layers Example**:
    - Input image has 3 different channels (RGB)
    - Apply a convolutional layer with 5 kernels, resulting in 5 feature maps
    - If the kernel is 2x2 and has 3 components (RGB), the kernel will be 2x2x3
    - If the image is grayscale, the kernel will be 2x2x1
    - **Example Architecture**: input -> convolution + ReLU (32 kernels) -> pooling (64 kernels) -> convolution + ReLU (128 kernels) -> pooling -> flattening -> fully connected layer -> softmax activation function -> output layer

- Multiple convolutional layers are needed to extract more complex features from the input data.
    - The first convolutional layers extract simple features like edges and colors, while deeper layers extract more complex features like shapes and patterns.
    - Examples of complex features are eyes, ears, and noses in facial recognition tasks.

- **Batch Normalization**: A technique that normalizes the input data to improve model performance. It helps the next layer learn more efficiently by providing normalized input data. It reduces the internal covariate shift by stabilizing the input distributions. Batch normalization is applied after the convolutional layer and before the activation function.

## Underfitting
- The model is too simple to capture the underlying patterns in the data.

## Model Capacity
- Improved by transfer learning, data augmentation, and hyperparameter tuning.
- The ability of the model to capture complex patterns in the data.

## Transfer Learning
- The process of using a pre-trained model to solve a similar problem.

### Strategies:
1. Train the entire model (better if you have a large dataset but different from the pre-trained model)
2. Train some layers and freeze others (better if you have a large dataset and similar to the pre-trained model)
3. Train some layers and freeze others (better if you have a small dataset and different from the pre-trained model)
4. Freeze the convolutional base (better if you have a small dataset and similar to the pre-trained model)

- **Example**: Image classification task with a pre-trained model on the ImageNet dataset.

- **Note**: Freezing layers, then training layers, and then freezing layers again is not recommended because the model will forget the learned patterns.

- **Fine-Tuning**: The process of training the pre-trained model on a new dataset to improve performance.

- Keep the convolutional base frozen if you have a small dataset to avoid overfitting.

## Difference Between Hyperparameter and Parameter
- **Hyperparameter**: Set before training the model (learning rate, batch size, number of epochs)
- **Parameter**: Learned during training (weights and biases)

## Data Augmentation
- The process of generating new training samples by applying random transformations to the existing data.

## Convolution to MLP Bridge
- Done by the flatten layer -> feature map to vector
- **Global Average Pooling Layer**: Reduces the spatial dimensions of the feature map to a single value.
    - Flattening: 16x16x4 -> 1024
    - Global Average Pooling: 16x16x4 -> 4
- Two ways to transform 2D data to 1D data.

## Recurrent Neural Networks (RNNs)
- Specialized neural networks for sequential data (data that has a temporal or sequential order).

- **RNN**: A type of NN that iterates over a sequence (of vectors) while keeping an internal state (memory) that depends on the previous elements of the sequence.

- **Vanishing Gradient Problem**: When the gradients become too small during backpropagation, making it hard for the model to learn long-term dependencies.

## Transformers
### Architecture:
- **Encoder**: Understands the input sequence and extracts its features
    - **Input Embedding**: Converts input tokens into vectors
        - For text: One-hot encoding (the number of "classes" is the size of the vocabulary)
        - Embedding (projection): Projects the one-hot encoded vectors into a lower-dimensional space that the model takes as input
    - **Positional Encoding**: Adds positional information to the input embeddings
    - **Self-Attention Mechanism**: Calculates the importance of each token in the sequence
        - The sequence elements are not aware of one another
        - The self-attention mechanism allows each element to consider the other elements in the sequence when making predictions.
        - Similarity scores are calculated between each pair of tokens in the sequence to determine their importance.
        - Example: x1 x2 x3 x4 x5
            - x1 is compared to x2, x3, x4, x5
            - x2 is compared to x1, x3, x4, x5...
            - It's a scalar product
        - Apply a softmax function to the similarity scores to get the attention weights (it's the score of the similarity)
        - The attention weights are multiplied by the input tokens to get the weighted sum, which is the output of the self-attention mechanism.
        - Now x1 = sum of (x1, x2, x3, x4, x5) * attention weights
        - Take each vector and put everything in matrix M
        - Multiply M transposed by M -> get the similarity matrix
        - Apply softmax to the similarity matrix -> get the attention matrix
        - Multiply the attention matrix by M -> get the output matrix
        - Add learnable weights to learn how to perform the self-attention mechanism:
            - The learnable weights are called query, key, and value matrices.
            - With x1 example -> query
            - x1(k1) x2(k2) x3(k3) x4(k4) ... xN(kN)
            - v1 v2 v3 v4 ... vN
            - x1 = softmax(query * key) * value
        - **Multi-Head Attention**: The self-attention mechanism is applied multiple times in parallel, each with different learnable weights.
    - **Feed Forward Neural Network**: Processes the self-attention output
        - Takes a matrix as input
        - **Residual Connection**: Adds the input to the feed-forward neural network output to prevent the vanishing gradient problem.
        - **Layer Normalization**: Normalizes the output of the residual connection to stabilize training.
        - Example: "The dog is sleeping and the cat is playing"
            - The model is aware that the cat is playing and the dog is sleeping because we add positional encoding to the input embeddings.
            - p(i,j) = sin(pos/10000^(2i/d)) if i is even
            - p(i,j) = cos(pos/10000^(2i/d)) if i is odd
            - pos: Position of the token in the sequence
            - i: Dimension of the embedding
            - d: Dimension of the embedding
        - The output of the encoder is a sequence of vectors that represent the input sequence's features.
    - **Layer Contains**: Input embedding -> positional encoding -> self-attention mechanism -> feed-forward neural network
        - The input embedding and positional encoding are not done each time we pass through the layer, they are done only once because they are not learnable parameters.
    - **Classification Task on a Sequence**:
        - At the end of the encoder, add a classification head that takes the encoder's output, which is a sequence of vectors.
        - Add a global average pooling layer to reduce the sequence of vectors to a single vector that is passed to the classification head.
        - Apply an MLP to the pooled vector to make predictions.

- **Decoder**: Generates the output sequence based on the encoder's features

## RNN vs Transformers
- RNN processes one by one, while Transformers process all in one.

## MLP GAN (Generative Adversarial Networks)
- A type of neural network architecture that consists of two networks: a generator and a discriminator.
    - **Generator**: Generates new samples from random noise
    - **Discriminator**: Distinguishes between real and generated samples
    - The generator tries to generate samples that are indistinguishable from real samples, while the discriminator tries to correctly classify the samples as real or generated.
    - The generator and discriminator are trained simultaneously in a min-max game, where the generator tries to fool the discriminator, and the discriminator tries to distinguish between real and generated samples.
    - The loss function of the GAN is the sum of the generator and discriminator losses.
    - The generator loss is the cross-entropy loss between the generated samples and the real samples, while the discriminator loss is the cross-entropy loss between the real and generated samples.
    - The GAN is trained using backpropagation and stochastic gradient descent.
    - The GAN architecture is used for generating new samples, such as images, text, and audio.
