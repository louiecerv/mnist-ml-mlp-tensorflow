#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

# Define the Streamlit app
def app():
    st.subheader('Neural Network Classifier Performance on the MNIST Digits Dataset')
    text = """Dataset: MNIST - 70,000 images of handwritten digits (28x28 pixels), each labeled 
    with its corresponding digit (0-9).
    \nThe performance of an MLP classifier on the MNIST handwritten digits dataset can be quite 
    good, typically achieving accuracy rates in the high 90s (often above 95%). """
    st.write(text)
    
    # Load MNIST dataset
    X, y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Define MLP parameters    
    st.sidebar.subheader('Set the MLP Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["lbfgs", "adam", "sgd"]
    solver = st.sidebar.selectbox('Select the solver:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=100,
        value=10,  # Initial value
        step=5
    )

    alpha = st.sidebar.slider(   
        label="Set the alpha:",
        min_value=.001,
        max_value=1.0,
        value=0.1,  # In1.0itial value
    )

    max_iter = st.sidebar.slider(   
        label="Set the max iterations:",
        min_value=100,
        max_value=500,
        value=300,  
        step=10
    )
    
    # Create MLPRegressor model
    clf = MLPClassifier(solver=solver, activation=activation, 
        alpha=0.01, hidden_layer_sizes=(hidden_layers, 10), 
        random_state=1,max_iter=max_iter)

    text = """Recommended ANN parameters: solver=lbfgs, activation=relu, n_hidden_layer=10, max_iter=500"""
    st.write(text)

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP classifier can take some time please wait...")

        # Train the model 
        clf.fit(X_train, y_train)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Classifier training completed!") 

        st.subheader('Performance of the MLP-ANN Classfier on the MNIST Digits Dataset')
        text = """We test the performance of the MLP Classifier using the 20% of the dataset that was
        set aside for testing. The classifier performance metrics are presented below."""
        st.write(text)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.subheader('Performance Metrics')
        st.write(classification_report(y_test, y_pred))  

#run the app
if __name__ == "__main__":
    app()
