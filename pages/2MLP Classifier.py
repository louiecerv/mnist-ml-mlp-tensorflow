#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
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
        value=100,  
        step=10
    )

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Define the MLP regressor model
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layers,5), 
            solver=solver, activation=activation, 
            max_iter=max_iter, random_state=42)

    text = """Recommended ANN parameters: solver=lbfgs, activation=relu, n_hidden_layer=150, max_iter=150"""
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

        st.subheader('Performance of the MLP-ANN Classifier on the Heart Disease Dataset')
        text = """We test the performance of the MLP Classifer using the 20% of the dataset that was
        set aside for testing. The confusion matrix and classification report are presented below."""
        st.write(text)

        # Make predictions on the test set
        y_test_pred = clf.predict(X_test)
        
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Performance test completed!") 
        
        st.subheader('Confusion Matrix')

        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)

        text = """The confusion matrix shows the performance of an MLP (Multi-Layer Perceptron) 
        classifier on a heart disease prediction task, classifying individuals as either 
        having or not having heart disease. Let's break down the results and their 
        implications for future unseen data:
        Cells (These numbers were produced by a run using certain parameters of the MLP.  Other 
        combinations of parameters could produce different values.):
        [79, 23]: This cell represents individuals correctly classified as healthy 
        (negative). There were 79 true negatives (TN).
        [11, 92]: This cell represents individuals with heart disease. The model 
        correctly classified 92 (true positives - TP) and misclassified 11 (false positives - FP) as healthy.
        Implications:
        Overall Accuracy: By adding TN and TP (79 + 92) and dividing by the total (205), 
        we get a baseline accuracy of (79 + 92) / 205 = 84.4%. This indicates the model 
        performs decently overall.
        False Positives (FP): The model incorrectly classified 11 individuals with heart 
        disease as healthy. This could be concerning in a heart disease prediction 
        scenario. A false positive might lead to someone with heart disease not 
        receiving proper treatment.
        True Negatives (TN): The model successfully identified 79 healthy individuals. 
        This is a positive aspect, meaning the model can avoid unnecessary interventions 
        for healthy people.
        Unseen Data: The accuracy on unseen data might be similar but can not be guaranteed. 
        The model's performance is based on the data it was trained on. Generalizability to 
        unseen data is a challenge in machine learning, and real-world data can deviate 
        from the training data in unforeseen ways.
        Overall: The model shows promise, but the false positives require further investigation.  
        Depending on the application,  mitigating these  false positives might be crucial.  
        It's important to consider additional metrics  like True Negatives Rate 
        (specificity) and False Positive Rate  (specificity) for a more comprehensive evaluation."""
        st.write(text)
        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

        text = """An accuracy of more than 80% on the heart disease dataset for the MLP classifier 
        indicates that the model performs well on the data it was trained on. It can correctly 
        identify 80% of the data points as having or not having heart disease based on the features 
        it was trained on.
        However, there are limitations to consider when interpreting this accuracy:
        Overfitting: The model might have memorized the training data too well and may not generalize
        well to unseen data. This means it might not perform as well on new heart disease cases it 
        has never encountered before.
        Data Representativeness: The accuracy reflects how well the model performs on the specific 
        dataset it was trained on. If the dataset doesn't represent the real-world distribution of 
        heart disease cases well, the 83% accuracy might not be reliable for real-world application.
        Therefore, based solely on the 83% accuracy, it's not possible to reliably predict unseen 
        heart disease data. 
        \nAccuracy is just one measure of performance. Consider metrics like precision, 
        recall, and F1 score to understand how well the model performs on different
        types of classification errors (false positives and false negatives).
        Domain knowledge: In the medical field, even a small misclassification 
        can have serious consequences. 
        \nConsult medical experts to understand the acceptable level of error 
        for this application."""
        
        with st.expander('Click to view more details.'):
            st.write(text)


#run the app
if __name__ == "__main__":
    app()
