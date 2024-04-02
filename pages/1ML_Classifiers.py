#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Define the Streamlit app
def app():
        
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

    # Extract only the specified number of images and labels
    size = 10000
    X, y = mnist
    X = X[:size]
    y = y[:size]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader('The task: Classify handwritten digits from 0 to 9 based on a given image.')
    text = """Dataset: MNIST - 70,000 images of handwritten digits (28x28 pixels), each labeled 
    with its corresponding digit (0-9).
    \nModels:
    \nK-Nearest Neighbors (KNN):
    \nEach image is represented as a 784-dimensional vector (28x28 pixels). 
    To classify a new image, its distance is measured to K nearest neighbors in the 
    training data. The majority class label among the neighbors is assigned to the new image.
    \nDecision Tree:
    \nA tree-like structure is built based on features (pixel intensities) of the images. 
    \nThe tree splits the data based on decision rules (e.g., "pixel intensity at 
    position X is greater than Y"). The new image is navigated through the tree based on 
    its features, reaching a leaf node representing the predicted digit class.
    \nRandom Forest:
    \nAn ensemble of multiple decision trees are built, each trained on a random subset of 
    features (pixels) and a random subset of data.
    \nTo classify a new image, it is passed through each decision tree, and the majority class 
    label from all predictions is assigned."""
    st.write(text)


    st.subheader('First 25 images in the MNIST dataset') 

    # Get the first 25 images and reshape them to 28x28 pixels
    train_images = np.array(X_train)
    train_labels = np.array(y_train)
    images = train_images[:25].reshape(-1, 28, 28)    
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot each image on a separate subplot
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i], cmap=plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Digit: {train_labels[i]}")
    # Show the plot
    plt.tight_layout()
    st.pyplot(fig)

    st.sidebar.subheader('Select the classifier')

    # Create the selecton of classifier

    clf = tree.DecisionTreeClassifier()    
    options = ['Decision Tree', 'Random Forest Classifier', 'Extreme Random Forest Classifier', 'K Nearest Neighbor']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Decision Tree':
`       clf = tree.DecisionTreeClassifier()
    elif selected_option =='Random Forest Classifier':
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
    elif selected_option=='Extreme Random Forest Classifier':
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
    elif selected_option == 'K Nearest Neighbor':
        clf = KNeighborsClassifier(n_neighbors=5)

    if st.button("Begin Training"):
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.write(cm)
        st.subheader('Performance Metrics')
        st.write(classification_report(y_test, y_test_pred))

        if selected_option =='Decision Tree':
            text = """Achieves good accuracy, but can be prone to 
            overfitting, leading to lower performance on unseen data.
            Simple and interpretable, allowing visualization of decision rules.
            Susceptible to changes in the training data, potentially 
            leading to high variance in predictions."""
        elif selected_option=='Extreme Random Forest Classifier':
            text = """Generally outperforms a single decision tree, 
            reaching accuracy close to 98%. Reduces overfitting through 
            averaging predictions from multiple trees. Ensemble method - 
            combines predictions from multiple decision trees, leading to 
            improved generalization and reduced variance. Less interpretable 
            compared to a single decision tree due to the complex 
            ensemble structure."""
        elif selected_option=='Extreme Random Forest Classifier':
            text = """Performance: Can achieve similar or slightly better 
            accuracy compared to a random forest, but results can vary 
            depending on hyperparameter tuning. Introduces additional randomness 
            during tree building by randomly selecting features at each split.  Aims to 
            further improve generalization and reduce overfitting by increasing 
            the diversity of trees in the ensemble. Requires careful 
            hyperparameter tuning to achieve optimal performance."""
            classifier = "Extreme Random Forest"
        elif selected_option == 'K Nearest Neighbor':
            text = """Accuracy: While KNN can achieve reasonable accuracy (around 80-90%), 
            it's often outperformed by more sophisticated models like Support 
            Vector Machines (SVMs) or Convolutional Neural Networks (CNNs) which can 
            reach over 97% accuracy.\nComputational cost: Classifying new data points 
            requires comparing them to all data points in the training set, making 
            it computationally expensive for large datasets like MNIST."""
        st.subheader('Performance of the ' + selected_option)
        with st.expander("Click to display more informqation.")
            st.write(text)

#run the app
if __name__ == "__main__":
    app()