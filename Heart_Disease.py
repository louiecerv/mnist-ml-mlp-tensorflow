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
    This app leverages a machine learning model to predict housing prices 
    based on various factors influencing the California housing market.
    Predict house prices using a trained MLP model. Explore the 
    influence of different features on the predicted price.
    Source: Derived from the 1990 U.S. Census data for California [1].
    Size: Contains 20,640 data points, each representing a census block group.
    Features:
    8 independent variables:
    MedInc: Median income in the block group.
    HouseAge: Median age of houses in the block group.
    AveRooms: Average number of rooms per household.
    AveBedrms: Average number of bedrooms per household.
    Population: Population of the block group.
    AveOccup: Average number of household members.
    Latitude: Geographical latitude of the block group centroid.
    Longitude: Geographical longitude of the block group centroid.
    Target variable:
    Median house value in dollars (scaled by dividing by 100,000).
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
