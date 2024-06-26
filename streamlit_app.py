import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
from modelo import model
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title='AUSMAR Prediction Model', page_icon='🦺')
st.title('🦺 AUSMAR SL - Stock Prediction Model')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to predict the amount of inventory needed for a specific reference for the following week.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and:\n'
               '1. Import current Purchase Orders\n'
               '2. Import current stock in ALM and ES\n'
               '3. Select one reference\n'
               'As a result, this will show the amount of units to purchase.')

    st.markdown('**Under the hood**')
    st.markdown('Data sets:')
    st.code('''
    - Current Purchase Orders
    - Current stock
    ''', language='markdown')

def predict(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction

# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1. Input data')

    st.markdown('**1. Use custom data**')

    uploaded_file_PO = st.file_uploader("Current Purchase Orders", type=["csv"])
    if uploaded_file_PO is not None:
        df_01 = pd.read_csv(uploaded_file_PO, index_col=False)

    uploaded_file_stock = st.file_uploader("Current Stock", type=["csv"])
    if uploaded_file_stock is not None:
        df_02 = pd.read_csv(uploaded_file_stock, index_col=False)

    st.header('2. Set Parameters')
    options = ['REF_001', 'REF_002', 'REF_003']
    reference = st.selectbox('Select one reference', options, index=0)
    date = st.date_input('Select date')
    sleep_time = st.slider('Sleep time', 0, 3, 0)

if st.button("Predict"):
    if (uploaded_file_PO is not None and uploaded_file_stock is not None) or True:
        with st.spinner("Running ..."):
            st.write("Loading data ...")
            time.sleep(sleep_time)

            st.write("Preparing data ...")
            time.sleep(sleep_time)

            X = df_01.iloc[:, :-1]
            y = df_01.iloc[:, -1]

            # Load the model from the file
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)

            st.write("Getting results ...")
            time.sleep(sleep_time)

            # Placeholder input data (this should be replaced with actual input collection logic)
            input_data = [1, 1, 1, 1]
            prediction = predict(input_data)

            # Display prediction results
            st.write(f"The predicted amount of units to purchase is: {prediction[0]}")

else:
    st.warning('👈 Please upload both Current Purchase Orders and Current Stock CSV files to proceed!')

#Estudio del modelo de Machine learning

with st.expander('ML Visualizer'):
    st.markdown('**Estudio descriptivo**')


    #Cargamos el archivo:
    #Air = pd.read_csv('data/AirPassengers.csv')

    #Convertimos a serie temporal:
    #Air.set_index(['Month'],inplace=True)
   # Air.index= pd.to_datetime(Air.index)

   # plt.rcParams["figure.figsize"] = (12, 10)
   # Air.plot()
   # plt.show()
