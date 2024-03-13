#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
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
  
        text = """Interpret the result."""

        st.write(text)



#run the app
if __name__ == "__main__":
    app()
