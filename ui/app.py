# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:04:13 2023

@author: abedn
"""

import numpy as np
import pandas as pd
from joblib import load
import streamlit as st
from sklearn.preprocessing import StandardScaler


model = load('../model/log_model.joblib')
sc =  StandardScaler()


### This is the function/method that handles the prediction
def prediction(age, IsActiveMember, balance):
    mean_data = [39.956604, 0.515100, 76485.889288]
    st_data = [10.487806, 0.499797, 62397.405202]
    
    
    age = (age - mean_data[0])/ st_data[0]
    IsActiveMember = (IsActiveMember - mean_data[1])/ st_data[1]
    balance = (balance - mean_data[2])/ st_data[2]
    prediction = model.predict(np.array([[age, IsActiveMember, balance]]))
    
    return prediction

# function to create the ui
def main():
    st.title("Bank Customer Churn Model")
    
    age = st.number_input('Enter your age: ')
    IsActiveMember = st.selectbox('Are you an active member? :', ['yes', 'no'])
    balance = st.number_input('Enter your balance: ')
    
    button =  st.button('Predict')
    
    if IsActiveMember.lower() == 'yes':
        IsActiveMember = 1
    else:
        IsActiveMember = 0
   
    result = ''
    
    if (button):
        result = prediction(age, IsActiveMember, balance)
        st.write('predicting.....')
        if result == 0:
            st.success('This user would not exit')
        else:
            st.success('This user would exit')
    
if __name__ == '__main__':
    main()


    



