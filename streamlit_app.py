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
from datetime import datetime, timedelta
import funciones as f

options = ['B062021', 'B062019', 'CA161497', 'CA161491', 'CA151153', 'B062007',
           'CA151162', 'CA161501', 'CA161509', 'CA140180', 'CA140181', 'CA030009',
           'CA030008', 'B062009', 'CA030010', 'PT', 'B051904', 'B051983',
           'B052000', 'CA161459', 'CA161621', 'CA150775', 'B992051', 'B051938',
           'B051936', 'B062047', 'TPLA1500000', 'CA161592', 'CA100118']

# Page title
st.set_page_config(page_title='AUSMAR Prediction Model', page_icon='⛵')
st.title('⛵ AUSMAR SL - Stock Prediction Model')

with st.expander('Información para el usuario'):
    st.markdown('**¿Cuál es la finalidad de esta herramienta?**')
    st.info('Esta herramienta ha sido diseñada con la finalidad de '
            'predecir la cantidad de inventario que se va a consumir en las próximas semanas')

    st.markdown('**¿Cómo utilizarla?**')
    st.info('Le recomendamos que se dirija al panel lateral izquierdo y siga los siguientes pasos:\n'
            '1. Asegúrese de que las bases de datos para entrenar el modelo están actualizada con los últimos datos de la semana pasada (recuadro en verde). \n'
            'En caso contrario, actualice los datos que faltan importando el archivo csv en "Añadir datos recientes". \n'
            'Si lo desea, puede actualizar por completo la base de datos en "Refrescar todos los datos". \n'
            '2. Una vez importados los datos, aparecerá un botón para cargar los datos "Load Data". Haga clic y asegúrese que el recuadro pasa de rojo a verde. \n'
            '3. Siempre que cargue nuevos datos es importante que vuelva a entrenar los modelos con el botón "Entrenar Modelos. \n'
            '4. Importe el csv del inventario actual disponible en "Inventario disponible" y las compras del último mes en "Pedidos recientes".\n'
            '5. Finalmente, seleccione los productos deseados para la predicción de consumos de las proximas semanas. \n')



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
    #current_date = datetime.date.today()
    current_date = pd.Timestamp(datetime.now())
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

    if st.button("Entrenar modelos"):
        try:
            f.update_models(options,"data/datos_entrenamiento_modelo.csv")
        except Exception as e:
            print(f"Error al entrenar modelos: {e}")
        st.success("Modelos entrenados con éxito")



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



    references = st.multiselect('Seleccione el producto', options)



    sleep_time = 0.5

if color == 'red':
    st.warning('👈 Porfavor, revise que los datos de aprendizaje estén actualizados. \n'
               'La precisión del modelo puede ser baja')
elif file_inventario is None or file_pedidos is None:
    st.warning('👈 No se han introducidos datos para el control de inventario y/o pedidos realizados. \n'
               'Las recomendaciones de compra pueden no ser precisas.')


st.markdown('### Datos para la predicción:')

input_ref = []
input_count = []


for r in references:
    input_datos = pd.read_csv("data/datos_entrenamiento_modelo.csv")
    input_ref.append(r)
    input_count.append(input_datos[input_datos["Producto"] == r]["Producto"].count())


val_entrada = {"Productos":input_ref,
               "Cantidad datos para entrenamiento": input_count
               }
st.table(val_entrada)

if st.button("Predict"):

    with st.status("Corriendo ...",expanded=True) as status:


        st.write("Cargando datos ...")
        time.sleep(0.5)
        inventario = pd.read_csv("data/inventario.csv")
        pedidos = pd.read_csv("data/pedidos.csv")

        #week_plus0 = current_date - pd.to_timedelta(current_date.dt.dayofweek, unit = 'D')
        week_plus0 = current_date - pd.to_timedelta(current_date.dayofweek, unit='D')
        week_plus1 = week_plus0 + timedelta(days=7)
        week_plus2 = week_plus1 + timedelta(days=7)
        week_plus3 = week_plus2 + timedelta(days=7)
        week_plus4 = week_plus3 + timedelta(days=7)
        week_plus0_str = week_plus0.strftime("%Y-%m-%d")
        week_plus1_str = week_plus1.strftime("%Y-%m-%d")
        week_plus2_str = week_plus2.strftime("%Y-%m-%d")
        week_plus3_str = week_plus3.strftime("%Y-%m-%d")
        week_plus4_str = week_plus4.strftime("%Y-%m-%d")


        st.write("Realizando predicciones ...")
        time.sleep(1)

        productos = []
        pred00 = []
        pred01 =[]
        pred02 = []
        pred03 =[]
        pred04 = []
        total = []
        recom = []
        inv = []
        ped = []
        Ljung = []
        mape =[]

        datos_plot = pd.read_csv("data/datos_entrenamiento_modelo.csv")
        for reference in references:

            datos_plot_ref = datos_plot[datos_plot["Producto"] == reference].groupby('Semana')['Cantidad'].sum().reset_index()
            datos_plot_ref.set_index(['Semana'], inplace=True)
            datos_plot_ref.index = pd.to_datetime(datos_plot_ref.index)
            tr = datos_plot_ref.iloc[:-20]
            tst = datos_plot_ref.iloc[-21:]



            ruta_modelo = f"modelos/hw_mul_model_{reference}.pkl"
            with open(ruta_modelo, 'rb') as file:
                model = pickle.load(file)

            inv_ref = inventario[inventario["Cod. Artículo"] == reference]["Stock unidades"].str.replace(',', '.').astype(
                float).sum().astype(int)
            ped_ref = pedidos[pedidos["Producto"] == reference]["Cantidad"].str.replace(',', '.').astype(float).sum().astype(int)
            prediction00 = model.predict(week_plus0_str)[0].round(0).astype(int)
            prediction01 = model.predict(week_plus1_str)[0].round(0).astype(int)
            prediction02 = model.predict(week_plus2_str)[0].round(0).astype(int)
            prediction03 = model.predict(week_plus3_str)[0].round(0).astype(int)
            prediction04 = model.predict(week_plus4_str)[0].round(0).astype(int)

            consumo_total = prediction00 + prediction01 + prediction02 + prediction03 + prediction04
            a_comprar =prediction00 + prediction01 + prediction02 + prediction03 + prediction04 - inv_ref - ped_ref
            if a_comprar < 0:
                a_comprar = 0



            ljung_box_p_value, mape_value = f.eval_model02(model, tr, tst, name=reference)

            productos.append(reference)
            pred00.append(prediction00)
            pred01.append(prediction01)
            pred02.append(prediction02)
            pred03.append(prediction03)
            pred04.append(prediction04)
            recom.append(a_comprar)
            total.append(consumo_total)
            inv.append(inv_ref)
            ped.append(ped_ref)
            Ljung.append(ljung_box_p_value)
            mape.append(mape_value.round(2))




        st.write("Obteniendo resultados ...")
        time.sleep(sleep_time)

    status.update(label="Hecho!", state="complete", expanded=False)

    results = {
        "Producto":productos,
        f"Consumos semana actual": pred00,
        f"Consumos semana {week_plus1.isocalendar()[1]}": pred01,
        f"Consumos semana {week_plus2.isocalendar()[1]}": pred02,
        f"Consumos semana {week_plus3.isocalendar()[1]}": pred03,
        f"Consumos semana {week_plus4.isocalendar()[1]}": pred04,
        "Consumo total (5 semanas)": total,
        "Inventario": inv,
        "Pedidos": ped,
        "Recom": recom,
        "pvalor": Ljung,
        "MAPE":mape
    }
    results_df = pd.DataFrame(results)


    # Mostrar resultado final
    st.success("Proceso de predicción completado correctamente.")

    # Mostrar los resultados finales


    # Display key-value pairs using Markdown
    st.write("### Tabla de resultados")
    st.table(results_df.iloc[:, :7])

    st.markdown("### Datos estadísticos del modelo:")


    for ref_plot in references:

        st.markdown(f"Artículo {ref_plot}")



        filtered_data = datos_plot[datos_plot["Producto"] == ref_plot].groupby("Semana")["Cantidad"].sum().reset_index()
        additional_points = pd.DataFrame({
            "Semana":[week_plus0,week_plus1,
                      week_plus2,week_plus3,
                      week_plus4],
            "Cantidad":results_df[results_df["Producto"]==ref_plot].iloc[0,1:6].to_list()
        })


        # Set the index to 'Semana' and convert to datetime
        filtered_data.set_index(['Semana'], inplace=True)
        additional_points.set_index(['Semana'], inplace = True)

        filtered_data.index = pd.to_datetime(filtered_data.index)
        additional_points.index = pd.to_datetime(additional_points.index)



        # Crear el gráfico de la línea principal
        line_chart = alt.Chart(filtered_data.reset_index()).mark_line().encode(
            x=alt.X('Semana:T', title='Semana'),
            y=alt.Y('Cantidad:Q', title='Cantidad')
        ).properties(
            width=800,
            height=250
        )

        # Crear los puntos adicionales en color naranja
        pred_chart = alt.Chart(additional_points.reset_index()).mark_line(color='orange').encode(
            x=alt.X('Semana:T', title='Semana'),
            y=alt.Y('Cantidad:Q', title='Cantidad')
        )

        # Superponer los puntos adicionales sobre el gráfico de línea
        combined_chart = line_chart + pred_chart



        #combined_chart.display()

        # Obtener el p-valor y MAPE para el producto ref_plot
        p_valor = results_df.loc[results_df['Producto'] == ref_plot, 'pvalor'].values[0].round(2)
        MAPE = results_df.loc[results_df['Producto'] == ref_plot, 'MAPE'].values[0].round(1) * 100
        inventario_disp = results_df.loc[results_df['Producto'] == ref_plot, 'Inventario'].values[0].round(2)
        pedidos02 = results_df.loc[results_df['Producto'] == ref_plot, 'Pedidos'].values[0].round(2)
        recomendacion = results_df.loc[results_df['Producto'] == ref_plot, 'Recom'].values[0].round(2)



        # Definir colores basados en condiciones
        if p_valor < 0.05:
            color01 = "red"
        else:
            color01 = "green"

        if 20 < MAPE < 50:
            color02 = "black"
        elif MAPE <= 20:
            color02 = "green"
        else:
            color02 = "red"

        # Mostrar en Markdown con colores condicionales
        st.markdown(
            f'<strong>Inventario actual:</strong> {inventario_disp} unidades\n'
            f'<strong>Pedidos realizado:</strong> {pedidos02} unidades\n',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<strong>Recomendación de compra:</strong> {recomendacion} unidades',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<strong>P-valor LjungBox:</strong> <span style="color:{color01}"> {p_valor}</span>\n'
            f'<strong>MAPE:</strong> <span style="color:{color02}"> {MAPE}%</span>\n',
            unsafe_allow_html=True
        )


        # Display the Altair chart in Streamlit
        st.altair_chart(combined_chart, use_container_width=True)


# Botón para generar el informe en PDF
  #  if st.button('Generar Informe PDF'):
  #      pdf_file = f.generate_pdf_report(results_df)
  #      st.download_button(label='Descargar PDF', data=pdf_file, file_name='informe.pdf', mime='application/pdf')



