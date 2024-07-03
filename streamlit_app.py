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
import funciones as f




# Page title
st.set_page_config(page_title='AUSMAR Prediction Model', page_icon='🦺')
st.title('🦺 AUSMAR SL - Stock Prediction Model')

with st.expander('Información para el usuario'):
    st.markdown('**¿Cuál es la finalidad de esta herramienta?**')
    st.info('Esta herramienta ha sido diseñada con la finalidad de '
            'predecir la cantidad de inventario que se va a consumir en las próximas semanas')

    st.markdown('**¿Cómo utilizarla?**')
    st.info('Le recomendamos que se diriga al panel lateral izquierdo y siga los siguientes pasos:\n'
            '1. Asegúrese que las base de datos está actualizada con los últimos datos de la semana anterior. \n'
            'En caso contrario, actualice los datos que faltan importando el archivo csv a través '
            'del botón "actualizar DB". \n'
            'Si lo desea, puede actualizar completamente la base de datos con la opción "Refrescar DB" \n'
            '2. Importe el csv de los ultimos datos de comprar del mes anterior\n'
            '3. Importe el csv de los niveles de inventario actualizados \n'
            '4. Seleccione la producto del que quiere realizar la modelización \n')



def predict(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction

# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1. Datos de aprendizaje para el modelo:')

    # Ejemplo de valores
    week_DB = f.week_number("data/datos_entrenamiento_modelo.csv")
    # Get the current date
    current_date = datetime.date.today()
    # Get the ISO calendar week number
    week_today = current_date.isocalendar()[1]

    # Condición para determinar el color
    if week_DB + 1 == week_today:
        color = 'green'
    else:
        color = 'red'

    with st.container(style=f"background-color: {color}; padding: 10px"):
        st.write(f"Últimos datos de la semana: {week_DB}")
        st.write(f"Semana actual: {week_today}")

    uploaded_file_PO = st.file_uploader("Current Purchase Orders", type=["csv"])
    if uploaded_file_PO is not None:
        PO = pd.read_csv(uploaded_file_PO, index_col=False)


    uploaded_file_stock = st.file_uploader("Current Stock", type=["csv"])
    if uploaded_file_stock is not None:
        stock = pd.read_csv(uploaded_file_stock, index_col=False)

    st.header('2. Set Parameters')
    options = ['REF001', 'REF002', 'REF003']
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

            prediction = model.predict(date)[0].round(0)
            #Falta añadir que se vaya sumando la prediccion de la semana actual + las siguintes hasta llegar a la semana de la cual se quiere predecir el stock
            

            if uploaded_file_PO is not None:
                PO_ref = PO[PO["reference"]==reference]["units"].sum()
            else:
                PO_ref = 0
            if uploaded_file_stock is not None:
                stock_ref = stock[stock["reference"]==reference]["units"].sum()
            else:
                stock_ref = 0



            st.write("Getting results ...")


            time.sleep(sleep_time)

            # Placeholder input data (this should be replaced with actual input collection logic)



            # Display prediction results
        status.update(label="Status", state="complete", expanded=False)

        st.write(f"For the reference: {reference} we have in total {stock_ref} units in stock.\n"
                 f"We have already purchased {PO_ref} units that will arrive soon. \n"
                 f"The total amount needed for the week {week_number} is {prediction} units \n"
                 f"So for this week {current_week} we recommend to purchase:\n"
                 f"{prediction} - {stock_ref} - {PO_ref} = {prediction - stock_ref - PO_ref} units")


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

