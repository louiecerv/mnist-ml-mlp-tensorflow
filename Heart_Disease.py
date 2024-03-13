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

    text = """Multi-Layer Perceptron Regressor on the California Housing Dataset"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('california.jpg', caption="California Housing Dataset")

    text = """
    This Streamlit app leverages an MLP classifier to predict the presence or absence of heart disease based on your 
    input of various heart symptoms. The data used to train the model comes from the heart disease dataset 
    [Kaggle Heart Disease Dataset].
    \nAnalyze your risk!
    Provide information on factors like chest pain, blood pressure, and cholesterol, 
    and this app will estimate your likelihood of having heart disease.
    \nDisclaimer: This app is for informational purposes only and should not be used for definitive medical diagnosis. 
    Please consult a healthcare professional for any concerns about your heart health.
    """
    st.text(text)

    text = """Scikit-learn's MLPRegressor is a tool for building multi-layer 
    perceptron (MLP) models for regression tasks. Unlike linear regression, 
    MLPs can model non-linear relationships between features and the target variable.
    
    \nFunction: Learns a non-linear mapping between input data and continuous target values.
    Architecture: Includes an input layer, one or more hidden layers with non-linear 
    activation functions, and an output layer.
    \nTraining: Optimizes the squared error using learning algorithms like LBFGS 
    or stochastic gradient descent.
    Uses: Suitable for complex regression problems where linear models might not 
    perform well."""

    st.write(text)


    
#run the app
if __name__ == "__main__":
    app()
