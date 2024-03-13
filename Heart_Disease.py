#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import time

# Define the Streamlit app
def app():
    if "X" not in st.session_state: 
        st.session_state.X = []
    
    if "y" not in st.session_state: 
        st.session_state.y = []

    if "scaler" not in st.session_state:
        st.session_state["scaler"] = StandardScaler()

    if "clf" not in st.session_state:
        st.session_state.clf = []

    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []

    if "X_test_scaled" not in st.session_state:
            st.session_state.X_test_scaled = []

    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 4

    text = """Multi-Layer Perceptron Classifier on the Heart Disease Dataset"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('heart-disease.jpg', caption="Heart Disease Diagnosis using MLP-ANN")

    text = """
    This Streamlit app leverages an MLP classifier to predict the presence or absence of 
    heart disease based on your input of various heart symptoms. The data used to train 
    the model comes from the heart disease dataset 
    [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset].
    \nAnalyze your risk!
    Provide information on factors like chest pain, blood pressure, and cholesterol, 
    and this app will estimate your likelihood of having heart disease.
    \nDisclaimer: This app is for informational purposes only and should not be used for definitive medical diagnosis. 
    Please consult a healthcare professional for any concerns about your heart health.
    """
    st.write(text)

    text = """
    An MLP (Multi-Layer Perceptron) classifier can be used to analyze a heart disease dataset and 
    predict whether a patient has heart disease or not. Here's how it works in this context:
    \nBinary Classification:
    The MLP aims for binary classification, meaning the output will be either 0 (no heart disease) or 1 (heart disease).
    \nData Preprocessing:
    The heart disease dataset contain various features like age, blood pressure, cholesterol levels, etc. 
    These features might need scaling or normalization for the MLP to process them efficiently.
    
    \nMLP Architecture:
    The MLP is a type of artificial neural network with an interconnected layer structure.
    In this case, the input layer will have the size matching the number of features in the heart 
    disease data (e.g., age, blood pressure).
    
    There will be one or more hidden layers containing a number of artificial neurons. 
    These hidden layers extract complex patterns from the data.  The final output layer will have 
    a single neuron with a sigmoid activation function. This neuron outputs a value between 0 and 1, 
    which is interpreted as the probability of having heart disease (closer to 1) or not (closer to 0).

    \nTraining:
    The MLP is trained using a labeled dataset where each data point has a confirmed classification 
    (heart disease or no disease) associated with its features.
    During training, the MLP adjusts the weights and biases between its artificial neurons to 
    minimize the error between its predicted probabilities and the actual labels in the training data.
    A common training algorithm for MLPs is backpropagation, which calculates the error and propagates 
    it backward through the network to update the weights and biases.
    \nPrediction:
    Once trained, the MLP can predict the probability of heart disease for new, unseen data points 
    based on their features. A threshold is typically set on the output probability (e.g., 0.5). 
    Values above the threshold are classified as having heart disease, while those below are 
    classified as healthy."""

    st.write(text)


    
#run the app
if __name__ == "__main__":
    app()
