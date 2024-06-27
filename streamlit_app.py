import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import altair as alt
import datetime as datetime
import datetime



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
        PO = pd.read_csv(uploaded_file_PO, index_col=False)

    uploaded_file_stock = st.file_uploader("Current Stock", type=["csv"])
    if uploaded_file_stock is not None:
        stock = pd.read_csv(uploaded_file_stock, index_col=False)

    st.header('2. Set Parameters')
    options = ['REF_001', 'REF_002', 'REF_003']
    reference = st.selectbox('Select one reference', options, index=0)
    date = st.date_input('Select date')
    today = datetime.datetime.now()
    current_week = today.isocalendar().week
    week_number = date.isocalendar().week
    st.write(f"We are at week: {current_week}")
    st.write(f"Prediction for week: {week_number}")
    sleep_time = 1

if st.button("Predict"):
    if (uploaded_file_PO is not None and uploaded_file_stock is not None) or True:

        with st.status("Running ...", expanded=True) as status:

            st.write("Loading data ...")
            time.sleep(sleep_time)


            st.write("Preparing data ...")
            time.sleep(sleep_time)

            # Load the model from the file
            with open('hw_mul_model.pkl', 'rb') as file:
                model = pickle.load(file)

            prediction = model.predict(date)

            if PO is not None:
                PO_ref = PO[reference].sum()
            else:
                PO_ref = 0
            if stock is not None:
                stock_ref = stock[reference].sum()
            else:
                sotck_ref = 0



            st.write("Getting results ...")


            time.sleep(sleep_time)

            # Placeholder input data (this should be replaced with actual input collection logic)



            # Display prediction results
        status.update(label="Status", state="complete", expanded=False)

        st.write(f"For the reference {reference},"
                 f"We have in total {stock_ref} units in several ES,"
                 f"And we have alrady purchased {PO_ref} units that will arrive"
                 f"the total amount needed for the week {week_number} is {prediction}"
                 f"So for this week {current_week} we recommend to purchase:"
                 f"{prediction} - {stock_ref} - {PO_ref} = {prediction - stock_ref - PO_ref} units to purchase")


else:
    st.warning('👈 Please upload both Current Purchase Orders and Current Stock CSV files to proceed!')

#Estudio del modelo de Machine learning

with st.expander('ML Visualizer'):
    st.header('Estudio Descriptivo', divider='rainbow')

    Air = pd.read_csv('data/AirPassengers.csv')
    Air.set_index(['Month'], inplace=True)
    Air.index = pd.to_datetime(Air.index)
    st.write(Air.head())

    # Plotting with Altair
    chart = alt.Chart(Air.reset_index()).mark_line().encode(
        x='Month:T',
        y='Passengers:Q'
    ).properties(
        width=800,
        height=400
    )

    # Display the Altair chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

