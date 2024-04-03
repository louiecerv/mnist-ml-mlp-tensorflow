#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import time

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


# Define the Streamlit app
def app():

    st.subheader('Tensorflow ANN on the MNIST Digit Dataset')
    text = """The performance of a TensorFlow Artificial Neural Network (ANN) on the 
    MNIST digits dataset depends on various factors including the architecture of 
    the neural network, the optimization algorithm used, the preprocessing of the
    data, and the hyperparameters chosen. However, typically ANNs perform quite well 
    on the MNIST dataset due to its relatively simple nature.
    \n1. **Accuracy**: With a well-designed neural network architecture and appropriate hyperparameters, it's common to achieve accuracy rates of over 95% on the MNIST dataset. In many cases, accuracies exceeding 98% are achievable, particularly with convolutional neural networks (CNNs) which are commonly used for image classification tasks like MNIST.
    \n2. **Training Time**: The training time of the ANN on the MNIST dataset depends on the complexity of the network architecture, the size of the dataset, and the computational resources available. Training ANNs on MNIST is generally quite fast, especially when using GPU acceleration.
    \n3. **Overfitting**: Since MNIST is a relatively small and simple dataset, overfitting can be a concern, especially with complex neural network architectures. Regularization techniques such as dropout and L2 regularization are commonly used to mitigate overfitting.
    \n4. **Hyperparameter Tuning**: Achieving optimal performance often requires tuning hyperparameters such as learning rate, batch size, number of hidden layers, number of neurons per layer, activation functions, etc. Techniques like grid search or random search are commonly employed to find the best combination of hyperparameters.
    \n5. **Visualization of Results**: It's common to visualize the performance of the ANN using confusion matrices, precision-recall curves, or ROC curves to gain insights into how well the model is performing for each class.
    \n6. **Generalization**: A well-trained ANN on the MNIST dataset should generalize well to unseen data, meaning it performs well on new, unseen digits beyond the training set.
    \nTensorFlow ANN's performance on the MNIST dataset is often quite impressive, especially given its simplicity, and it serves as a good starting point for learning and experimenting with neural networks and image classification tasks."""
    st.write(text)

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

   # Define ANN parameters    
    st.sidebar.subheader('Set the Neural Network Parameters')
    options = ["relu", "leaky_relu",  "tanh", "elu", "selu"]
    h_activation = st.sidebar.selectbox('Activation function for the hidden layer:', options)

    options = ["sigmoid", "softmax"]
    o_activation = st.sidebar.selectbox('Activation function for the output layer:', options)

    options = ["adam", "adagrad", "sgd"]
    optimizer = st.sidebar.selectbox('Select the optimizer:', options)

    n_layers = st.sidebar.slider(      
        label="Number of Neurons in the Hidden Layer:",
        min_value=16,
        max_value=128,
        value=64,  # Initial value
        step=16
    )

    epochs = st.sidebar.slider(   
        label="Set the number epochs:",
        min_value=5,
        max_value=50,
        value=10,
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=h_activation),
        tf.keras.layers.Dense(10, activation = o_activation)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), 'accuracy'],
    )

    with st.expander("CLick to display guide on how to select parameters"):
        text = """ReLU (Rectified Linear Unit): This is the most common activation function used 
        in convolutional neural networks (CNNs) for hidden layers. It outputs the input 
        directly if it's positive (f(x) = x for x >= 0) and sets negative inputs to zero 
        (f(x) = 0 for x < 0). ReLU is computationally efficient, avoids the vanishing 
        gradient problem, and often leads to good performance in CNNs.
        \nSigmoid: This activation function squashes the input values between 0 and 1 
        (f(x) = 1 / (1 + exp(-x))). It's typically used in the output layer of a CNN for 
        tasks like binary classification (predicting one of two classes). 
        However, sigmoid can suffer from vanishing gradients in deep networks.
        \nAdditional Activation Function Options for Hidden Layers:
        \nLeaky ReLU: A variant of ReLU that addresses the "dying ReLU" problem where some 
        neurons might never fire due to negative inputs always being zeroed out. 
        Leaky ReLU allows a small, non-zero gradient for negative inputs 
        (f(x) = max(α * x, x) for a small α > 0). This can help prevent neurons from 
        getting stuck and improve training.
        TanH (Hyperbolic Tangent): Similar to sigmoid, TanH squashes values 
        between -1 and 1 (f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))). 
        It can sometimes be more effective than sigmoid in certain tasks due to 
        its centered output range.
        \nChoosing the Right Activation Function:
        \nThe best activation function often depends on the specific problem and 
        network architecture. Here's a general guideline:
        \nHidden Layers: ReLU is a strong default choice due to its efficiency and 
        ability to avoid vanishing gradients. Leaky ReLU can be a good alternative, 
        especially in deeper networks. TanH is also an option, but ReLU is often preferred.
        \nOutput Layer:
        \nBinary Classification: Sigmoid is commonly used here for its ability to output 
        probabilities between 0 and 1.
        \nMulti-class Classification: In this case, you'd likely use a softmax activation 
        function in the output layer, which normalizes the outputs to probabilities that 
        sum to 1 (useful for predicting one of multiple exclusive classes).
        \nExperimentation:
        \nIt's always recommended to experiment with different activation functions to see 
        what works best for your specific CNN and dataset. You can try replacing "relu" 
        with "leaky_relu" or "tanh" in the hidden layers and "sigmoid" with "softmax" 
        in the output layer to see if it improves performance.
        \nBy understanding these activation functions and their trade-offs, you can 
        make informed choices to optimize your ANN for the task at hand."""
        st.write(text)

    if st.button('Start Training'):
 
        progress_bar = st.progress(0, text="Training the model please wait...")

        history = model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_test,
            callbacks=[CustomCallback()]
        )

        # Evaluate the model on the test data
        accuracy = model.evaluate(ds_test)
        print("Test accuracy:", accuracy)

        # Extract loss and accuracy values from history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        # Create the figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize for better visualization

        # Plot loss on the first subplot (ax1)
        ax1.plot(train_loss, label='Training Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy on the second subplot (ax2)
        ax2.plot(train_acc, 'g--', label='Training Accuracy')
        ax2.plot(val_acc, 'r--', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Set the main title (optional)
        fig.suptitle('Training and Validation Performance')

        plt.tight_layout()  # Adjust spacing between subplots
        st.pyplot(fig)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Classifier training completed!") 

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.text(f"Epoch {epoch}: loss = {loss:.4f} accuracy = {accuracy:.4f}")

#run the app
if __name__ == "__main__":
    app()
