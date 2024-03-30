#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Define the Streamlit app
def app():
    if "dataset_ready" not in st.session_state:
        st.error("Dataset must be loaded. Click Heart Disease in the sidebar.")

    text = """The goal is to predict whether a patient has heart disease 
    (positive class) or not (negative class) based on various features collected 
    about their health. This is a binary classification task because the model 
    predicts one of two possible outcomes.
    \nDataset:
    The heart disease dataset is a popular benchmark dataset used in machine learning 
    for classification tasks. It contains information about patients, including features 
    like age, blood pressure, cholesterol levels, and heart rate. The target variable 
    indicates the presence or absence of heart disease.
    
    \nMLP Classifier:
    Scikit-learn's MLP Classifier is a Multi-Layer Perceptron, a type of artificial 
    neural network. In this scenario, the MLP is trained to learn the complex relationships 
    between the patient's features and the presence of heart disease. The model learns 
    through hidden layers of interconnected nodes, allowing it to capture non-linear 
    patterns in the data.
    \nProcess:
    Data Preprocessing: The heart disease data might require preprocessing steps like handling missing values and scaling the features to ensure a consistent range for the neural network.
    Model Training: The MLP classifier is trained on a portion of the data. During training, the model adjusts its internal weights and biases to minimize the error between its predictions and the actual presence or absence of heart disease for each patient.
    Evaluation: The performance of the trained model is evaluated on a separate hold-out test set. Metrics like accuracy, precision, recall, and F1-score can be used to assess how well the model generalizes to unseen data.
    By effectively using the MLP classifier on the heart disease dataset, you can build a 
    model that can predict the likelihood of heart disease in new patients, aiding in 
    early diagnosis and preventative measures."""
    st.write(text)
    
   # Define MLP parameters    
    st.sidebar.subheader('Set the MLP Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["lbfgs", "adam", "sgd"]
    solver = st.sidebar.selectbox('Select the solver:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=250,
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
        max_value=300,
        value=500,  
        step=10
    )

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Create MLPRegressor model
    clf = MLPRegressor(solver=solver, activation=activation, 
        alpha=0.01, hidden_layer_sizes=(hidden_layers, 10), 
        random_state=1,max_iter=max_iter)

    text = """Recommended ANN parameters: solver=lbfgs, activation=relu, n_hidden_layer=10, max_iter=500"""
    st.write(text)

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP regressor can take some time please wait...")

        # Train the model 
        clf.fit(X_train, y_train)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Regressor training completed!") 

        st.subheader('Performance of the MLP-ANN Regressor on the Advertising Dataset')
        text = """We test the performance of the MLP Regressor using the 20% of the dataset that was
        set aside for testing. The regressor performance metrics are presented below."""
        st.write(text)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Performance test completed!") 
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print performance metrics
        st.subheader('Performance Metrics')
        st.write("Mean Squared Error: {:.4f}".format(mse))  
        st.write("R2 score: {:.4f}".format(r2))


#run the app
if __name__ == "__main__":
    app()
