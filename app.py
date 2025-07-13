import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('iris_prediction.pkl','rb'))
st.title("Customer Segmentation")

petal_len = float(st.text_input("petal_len: "))
petal_wid = float(st.text_input("petal_wid: "))
sepal_len = float(st.text_input("sepal_len: "))
sepal_wid = float(st.text_input("sepal_wid: "))

featureInput = np.array([[petal_len,petal_wid,sepal_len,sepal_wid]])

sal = model.predict(featureInput)

st.write(f'Hello, *World!* :sunglasses: . Customer Group : {sal} ')
