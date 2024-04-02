#Input the relevant libraries
import streamlit as st

import time

# Define the Streamlit app
def app():

    text = """Three-way comparison of ML Classifiers, MLP and Tensorflow Artificial Neural Networks on the Advertising Dataset"""
    st.subheader(text)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('MNIST.png', caption="Modified National Institute of Standards and Technology", use_column_width=True)

    text = """MNIST is a large database of handwritten digits that is commonly used for training and
    testing various image processing systems. The acronym stands for Modified National Institute 
    of Standards and Technology. MNIST is a popular dataset in the field of machine learning and 
    can provide a baseline for benchmarking algorithms."""
    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
