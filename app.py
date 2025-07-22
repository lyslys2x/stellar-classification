import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

model = joblib.load("stellar-classification.pkl")

st.title("Stellar Object Classifier")

#html_file ="index.html"
#css_file = "style.css"
# Inject custom CSS
#def load_css(file_name):
#   with open(file_name) as f:
#        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject custom HTML#
# def load_html(file_name):
    #with open(file_name, 'r') as f:
        #html = f.read()
        #st.markdown(html, unsafe_allow_html=True)

# calling input/form elements
# Using number inputs to control the range of value the user is entering
alpha_selected = st.number_input("Enter Alpha", min_value=0, max_value=120, step=1)
delta_selected = st.number_input("Enter Delta", min_value=0, max_value=120, step=1)
red_shift_selected = st.number_input("Enter Red Shift", min_value=0, max_value=120, step=1)
green_filter_selected = st.number_input("Enter Green Filter", min_value=0, max_value=120, step=1)
red_filter_selected = st.number_input("Enter Red Filter", min_value=0, max_value=120, step=1)
ir_filter_selected = st.number_input("Enter Infrared Filter", min_value=0, max_value=120, step=1)
near_ir_filter_selected = st.number_input("Enter Near Infrared Filter", min_value=0, max_value=120, step=1)

# Check if inputs are empty before entering them into the model

# Predict button

if st.button("Prdict Stellar Object"):

    input_data = {
        "alpha": alpha_selected,
        "delta": delta_selected,
        "red_shift": red_shift_selected,
        "green_filter": green_filter_selected,
        "red_filter": red_filter_selected,
        "IR_filter": ir_filter_selected,
        "near_IR_filter": near_ir_filter_selected,
    }

    transformed_input = Pipeline.transform(input_data)

    df_input = pd.DataFrame([transformed_input])

    df_input = df_input.reindex(columns= model.feature_names_in_, fill_value=0)
    y_unseen_pred = Pipeline.predict(df_input)[0]
    st.success(f"Predicted Stellar Object: ${y_unseen_pred:, .2f}")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("../images/background.png");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html= True
)