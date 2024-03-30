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
    if "dataset_ready" not in st.session_state:
        st.error("Dataset must be loaded. Click Heart Disease in the sidebar.")
        
    st.subheader("Heart Disease Classification as a Binary Task")
    text = """In a heart disease classification task, the objective is to build 
    a model that can predict whether a patient is likely to have heart 
    disease based on a set of features. This is a classic example of a 
    binary classification problem. The model is trained on a dataset where 
    each data point represents a patient, and includes features like age, 
    blood pressure, cholesterol levels, etc. Each data point also has a 
    corresponding label indicating the presence (positive) or absence 
    (negative) of heart disease.  
    \nThe goal of the model is to learn the relationship between these features 
    and the disease, and then use that knowledge to classify new, 
    unseen patients as belonging to either the "heart disease" or 
    "no heart disease" category.
    \nClassifiers chosen:
    K-Nearest Neighbors (KNN): This algorithm classifies a data point by 
    identifying the k nearest data points in the training set based on 
    feature similarity. The majority class among these k neighbors is 
    then assigned as the predicted class for the new data point. KNN is a 
    simple and interpretable method, but can be computationally expensive 
    for large datasets. 
    \nSupport Vector Machine (SVM):  This algorithm creates a hyperplane in the 
    feature space that best separates the data points belonging to different classes. 
    New data points are then classified based on which side of the hyperplane 
    they fall on. SVMs are known for good performance on various classification 
    tasks, but can be sensitive to parameter tuning.
    \nNaive Bayes:** This probabilistic classifier uses Bayes' theorem to calculate 
    the probability of a data point belonging to a particular class based on its 
    features. It assumes independence between features, which might not always 
    hold true in real-world data. However, Naive Bayes is a fast and efficient 
    classifier that can be effective for certain problems.
    \nThese three algorithms represent a good selection of common machine 
    learning approaches for binary classification tasks like heart disease 
    prediction. Each has its own strengths and weaknesses, and the best choice for a
    specific problem can depend on the characteristics of the data and the desired 
    model properties. """
    st.write(text)

    #add the classifier selection to the sidebar
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

        st.subheader('Performance of ' + classifier)
        st.write(text)

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