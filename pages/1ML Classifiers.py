#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
import time

# Define the Streamlit app
def app():

    #add the classfier selection to the sidebar

    clf = KNeighborsClassifier(n_neighbors=5)
    options = ['K Nearest Neighbor', 'Support Vector Machine', 'Naive Bayes']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Support Vector Machine':
        clf = SVC(kernel='linear')
        st.session_state['selected_model'] = 1
    elif selected_option=='Naive Bayes':        
        clf = GaussianNB()
        st.session_state['selected_model'] = 2
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 0

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    
    st.subheader('Performance of ' + classifier)
    st.write(text)

    if st.button("Begin Training"):

        if selected_option =='K Nearest Neighbor':
            text = """KNN achieves good accuracy on the heart disease dataset, often 
            reaching around 85-90%. However, it can be slow for large datasets 
            due to needing to compare each test image to all training images. 
            Additionally, choosing the optimal number of neighbors (k) can be 
            crucial for performance."""
            classifier = 'K-Nearest Neighbor'
        elif st.session_state['selected_model'] == 1:   # SVM
            text = """SVM can also achieve high accuracy on this dataset, 
            similar to KNN. It offers advantages like being memory-efficient, 
            but choosing the right kernel function and its parameters 
            can be challenging."""
            classifier = 'Support Vector Machine'
        elif selected_option=='Naive Bayes': 
            text = """Naive Bayes is generally faster than the other two options but 
            may achieve slightly lower accuracy, typically around 80-85%. It performs 
            well when the features are independent, which might not perfectly hold true 
            for data found in the heart disease dataset."""
            classifier = "Naive Bayes"
                    
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))







#run the app
if __name__ == "__main__":
    app()