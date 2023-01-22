import pickle
from statistics import mode
# from tkinter.tix import Tree
from turtle import shape
import streamlit as st
from Main import vector_converter,clean_lammatize


# loading developed model
file = open("C:/Users/Asus/Documents/Sentiment_Model.pkl","rb")
model = pickle.load(file)


st.title("Sentimental Analysis On Food Review")



l = []

review = st.text_input(label="Enter review to sentiment")
l.append(review)
cleaned_review = clean_lammatize(l)
vector_review = vector_converter(cleaned_review)
btn = st.button(label="Sentiment")
# if btn pressed then predict
if btn == True:
    sentiment = model.predict(vector_review)
    if sentiment ==1 :
        st.success("Review is Positive")
    else:
        st.error("Review is Negative")
