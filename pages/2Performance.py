#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Define the Streamlit app
def app():
    st.subheader('Performance of the MLP-ANN Classifier on the Heart Disease Dataset')
    text = """We test the performance of the MLP Classifer using the 20% of the dataset that was
    set aside for testing. The confusion matrix and classification report are presented below."""
    st.write(text)
    
    if st.button('Begin Test'):
        
        X_test_scaled = st.session_state.X_test_scaled
        # Make predictions on the test set
        y_test_pred = st.session_state.clf.predict(X_test_scaled)
        y_test = st.session_state.y_test

        st.subheader('Confusion Matrix')

        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
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
        heart disease data. Here's what you can do for a more reliable assessment:
        Test on unseen data: Split the data into training and testing sets. Train the model on the 
        training set and evaluate its performance on the unseen testing set. This will give a better 
        idea of how well it generalizes.
        Look at other metrics: Accuracy is just one measure of performance. Consider metrics 
        like precision, recall, and F1 score to understand how well the model performs on different
        types of classification errors (false positives and false negatives).
        Domain knowledge: In the medical field, even a small misclassification can have serious 
        consequences. Consult medical experts to understand the acceptable level of error 
        for this application."""

        st.write(text)



#run the app
if __name__ == "__main__":
    app()
