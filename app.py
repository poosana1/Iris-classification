import streamlit as st
import pickle
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
from PIL import Image

pickle_in = open('Iris_model.pkl', 'rb')
classifier = pickle.load(pickle_in)


def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return prediction

def Input_Output():
    st.title("Iris Variety Prediction")
    st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png", width=600)

    st.markdown("You are using Streamlit...", unsafe_allow_html=True)
    sepal_length = st.slider("Enter Sepal Length", 0.0, 10.0)
    sepal_width = st.slider("Enter Sepal Width", 0.0, 10.0)
    petal_length = st.slider("Enter Petal Length", 0.0, 10.0)
    petal_width = st.slider("Enter Petal Width", 0.0, 10.0)

    result = predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width)
    if st.button("Click here to Predict"):
        result = predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width)
        st.balloons()
    st.success('The output is {}'.format(result))

if __name__ == '__main__': 
    Input_Output()
 