## What to remember:

* np.exp(x) works for any np.array x and applies the exponential function to every coordinate
* the sigmoid function and its gradient
* image2vector is commonly used in deep learning
* np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward 
eliminating a lot of bugs.
* numpy has efficient built-in functions
* broadcasting is extremely useful
* Vectorization is very important in deep learning. It provides computational efficiency and clarity.

* Common steps for pre-processing a new dataset are:

    * Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    * Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
    * "Standardize" the data



## Activation Function
Activation functions are really important for a Artificial Neural Network to learn and make sense of something really complicated and Non-linear complex functional mappings between the inputs and response variable.They introduce non-linear properties to our Network.Their main purpose is to convert a input signal of a node in a A-NN to an output signal. That output signal now is used as a input in the next layer in the stack.

Specifically in A-NN we do the sum of products of inputs(X) and their corresponding Weights(W) and apply a Activation function f(x) to it to get the output of that layer and feed it as an input to the next layer.