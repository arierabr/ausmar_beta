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
    st.info('Le recomendamos que se dirija al panel lateral izquierdo y siga los siguientes pasos:\n'
            '1. Asegúrese de que las bases de datos para entrenar el modelo están actualizada con los últimos datos de la semana pasada. \n'
            'En caso contrario, actualice los datos que faltan importando el archivo csv en "Añadir datos recientes". \n'
            'Si lo desea, puede actualizar por completo la base de datos en "Refrescar todos los datos". \n'
            '2. Importe el csv de los datos de compra del último mes en "Pedidos recientes". \n'
            '3. Importe el csv del inventario actual disponible en "Inventario disponible". \n'
            '3. Seleccione los productos deseados para la predicción de consumos en las proximas dos semana. \n')



#def predict(input_data):
#    input_array = np.array(input_data).reshape(1, -1)
#    prediction = model.predict(input_array)
#    return prediction

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
        color = '#8FBC8F'
    else:
        color = 'red'

    st.markdown(f'<div style="background-color: {color}; padding: 10px;">'
                f'<p>Semana de actualización: {week_DB}</p>'
                f'<p>Semana actual: {week_today}</p>'
                '</div>', unsafe_allow_html=True)


    new_data = st.file_uploader("Añadir datos recientes", type=["csv"])
    if new_data is not None:
        df = f.update_df("data/datos_entrenamiento_modelo.csv", new_data)
        st.success("Archivo importado con éxito")


    df_all_data = st.file_uploader("Refrescar todos los datos", type=["csv"])
    if df_all_data is not None:
        df = f.import_data(df_all_data)
        st.success("Archivo importado con éxito")

    if (df_all_data is not None) or (new_data is not None):
        if st.button("Load Data"):
            try:
                f.load_data(df)
            except Exception as e:
                print(f"Error al cargar los datos: {e}")


    st.header('2. Importar pedidos e inventarios')

    # Subir archivo de pedidos
    file_pedidos = st.file_uploader("Pedidos recientes", type=["csv"])
    if file_pedidos is not None:
        f.update_pedidos(file_pedidos)
        st.success("Archivo de pedidos importado con éxito")

    file_inventario = st.file_uploader("Inventario disponible", type=["csv"])
    if file_inventario is not None:
        f.update_stock(file_inventario)
        st.success("Archivo de inventario importado con éxito")


    st.header('3. Seleccionar producto')

    options = ["CA140180","CA140181","CA030009","CA030010","CA161459","CA030008","CA100118"]
    reference = st.selectbox('Seleccione el producto', options, index=0)


    sleep_time = 0.5

if color == 'red':
    st.warning('👈 Porfavor, revise que los datos de aprendizaje estén actualizados, \n'
               'La precisión del modelo puede ser baja')
elif file_inventario is None or file_pedidos is None:
    st.warning('👈 No se han introducidos datos para el control de inventario y/o pedidos realizados. \n'
               'Las recomendaciones de compra pueden no ser precisas.')




if st.button("Predict"):

    with st.spinner("Corriendo ..."):
        # Ejecutar el proceso de predicción y carga de datos
        f.update_models(options, "data/datos_entrenamiento_modelo.csv")

        st.write("Cargando datos ...")
        time.sleep(sleep_time)

        ruta_modelo = f"modelos/hw_mul_model_{reference}.pkl"
        inventario = pd.read_csv("data/inventario.csv")
        pedidos = pd.read_csv("data/pedidos.csv")
        week_plus1 = current_date + datetime.timedelta(days=7)
        week_plus2 = current_date + datetime.timedelta(days=14)
        week_plus1_str = week_plus1.strftime("%Y-%m-%d")
        week_plus2_str = week_plus2.strftime("%Y-%m-%d")

        st.write("Preparando modelo ...")
        time.sleep(sleep_time)

        with open(ruta_modelo, 'rb') as file:
            model = pickle.load(file)

        inv_ref = inventario[inventario["Cod. Artículo"] == reference]["Stock unidades"].str.replace(',', '.').astype(
            float).sum().astype(int)
        ped_ref = pedidos[pedidos["Producto"] == reference]["Cantidad"].str.replace(',', '.').astype(float).sum().astype(int)

        st.write("Realizando predicciones ...")
        time.sleep(sleep_time)

        prediction01 = model.predict(week_plus1_str)[0].round(0).astype(int)
        prediction02 = model.predict(week_plus2_str)[0].round(0).astype(int)
        total = prediction01 + prediction02 - inv_ref - ped_ref
        if total < 0:
            total = 0

        st.write("Obteniendo resultados ...")
        time.sleep(sleep_time)

        # Mostrar resultado final
        st.success("Proceso de predicción completado correctamente.")

        # Mostrar los resultados finales

        data = {
            f"Consumos semana {week_today + 1}": prediction01,
            f"Consumos semana {week_today + 2}": prediction02,
            "Stock disponible": inv_ref,
            "Pedidos por llegar": ped_ref,
            "Recomendación de compra": total
        }
        # Display key-value pairs using Markdown
        st.write("### Tabla de resultados")
        for key, value in data.items():
            st.write(f"- **{key}**:   {value} unidades")

        st.markdown("### Datos estadísticos del modelo")

        consumos = pd.read_csv('data/datos_entrenamiento_modelo.csv')

        consumos = consumos[consumos["Producto"] == reference].groupby("Semana")["Cantidad"].sum().reset_index()

        consumos.set_index(['Semana'], inplace=True)
        consumos.index = pd.to_datetime(consumos.index)
        st.table(consumos.tail())

        # Plotting with Altair
        chart = alt.Chart(consumos.reset_index()).mark_line().encode(
            x='Semana:T',
            y='Cantidad:Q'
        ).properties(
            width=800,
            height=400
        )

        # Display the Altair chart in Streamlit
        st.altair_chart(chart, use_container_width=True)





