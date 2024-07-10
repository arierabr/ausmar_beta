import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pickle
import os
import subprocess
import git





def eval_model(model,tr,tst,name='Model',lags=12):
    lb = np.mean(sm.stats.acorr_ljungbox(model.resid, lags=lags, return_df=True).lb_pvalue)
    pred = model.forecast(steps=len(tst))
    fig1, ax = plt.subplots()
    ax.plot(tr, label='training')
    ax.plot(tst, label='test')
    ax.plot(pred, label='prediction')
    plt.legend(loc='upper left')
    tit = name + ":  LjungBox p-value --> " + str(lb) + "\n MAPE: " + str(round(mean_absolute_percentage_error(tst, pred)*100,2)) + "%"
    plt.title(tit)
    plt.ylabel('Cantidad')
    plt.xlabel('Date')
    plt.show()
    print(lb)


def eval_model02(model, tr, tst, name='Model', lags=12):
    lb = np.mean(sm.stats.acorr_ljungbox(model.resid, lags=lags, return_df=True).lb_pvalue)
    pred = model.forecast(steps=len(tst))

    fig1, ax = plt.subplots()
    ax.plot(tr, label='training')
    ax.plot(tst, label='test')
    ax.plot(pred, label='prediction')
    plt.legend(loc='upper left')
    tit = name + ":  LjungBox p-value --> " + str(lb) + "\n MAPE: " + str(
        round(mean_absolute_percentage_error(tst, pred) * 100, 2)) + "%"
    plt.title(tit)
    plt.ylabel('Cantidad')
    plt.xlabel('Date')

    plt.close(fig1)  # Close the plot to prevent it from displaying automatically

    return lb, mean_absolute_percentage_error(tst, pred)

def update_models(lista_referencias, data_path):
    df = pd.read_csv(data_path)
    for referencia in lista_referencias:
        # Filter the DataFrame for the current reference and group by week
        df_producto = df[df["Producto"] == referencia].groupby("Semana")["Cantidad"].sum().reset_index()
        # Set 'Semana' as the index and ensure it's in datetime format
        df_producto.set_index('Semana', inplace=True)
        df_producto.index = pd.to_datetime(df_producto.index)
        # Crear un rango completo de fechas semanales
        full_range = pd.date_range(start=df_producto.index.min(), end=df_producto.index.max(), freq='W-MON')
        df_producto_new = df_producto.reindex(full_range, fill_value=0.1)


        # Fit the ExponentialSmoothing model
        hw_mul = ets.ExponentialSmoothing(df_producto_new, trend='mul', damped_trend=False, seasonal='mul', seasonal_periods= 54).fit()

        # Save the model to a file
        model_path = f'modelos/hw_mul_model_{referencia}.pkl'
        with open(model_path, 'wb') as file:
            pickle.dump(hw_mul, file)

        # Usar GitPython para hacer commit y push al repositorio de GitHub
        repo = git.Repo('.')
        repo.git.add(model_path)
        repo.index.commit('Actualizar modelos')

        # Obtener el token de acceso personal desde la variable de entorno
        token = os.getenv('GITHUB_TOKEN')
        if token is None:
            raise ValueError("El token de GitHub no está configurado como variable de entorno.")

        # Configurar la URL remota con el token
        repo_url = f"https://{token}@github.com/arierabr/ausmar_beta.git"  # Reemplaza 'usuario' y 'repo' con tus valores
        origin = repo.remote(name='origin')
        origin.set_url(repo_url)

        # Hacer push al repositorio remoto
        origin.push()
        print("Archivo subido al repositorio de GitHub")

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def update_df(data_path, new_csv):
    df = pd.read_csv(data_path)
    df_new = pd.read_csv(new_csv, encoding="latin1", header=0, sep=";")
    # Corregimos el nombre de las columnas:
    df_new.columns = [
        'Fecha', 'Dia semana', 'Estado doc.', 'Almacen', 'Producto', 'Subfamilia',
        'Nº productos', 'Nº Documentos', 'Nº items (lineas)', 'Cantidad', 'Importe bruto (euro)',
        'Importe dto. (euro)', '% Dto. Efectivo', 'Importe (euro)', 'Margen Efectivo (euro)',
        'Coste total (euro)', '% Margen Efectivo', 'Unnamed: 17']

    # Convertimos la columna 'Fecha' a tipo datetime
    df_new['Fecha'] = pd.to_datetime(df_new['Fecha'], format="%d/%m/%Y")
    df_new['Cantidad'] = df_new['Cantidad'].str.replace(',', '.')
    df_new['Cantidad'] = df_new['Cantidad'].astype('float')

    # Ajustamos las fechas para que el día sea el primer día de la semana (lunes)
    df_new['Semana'] = df_new['Fecha'] - pd.to_timedelta(df_new['Fecha'].dt.dayofweek, unit='D')

    # Pasamos todos los valores a absoluto:
    df_new["Cantidad"] = df_new["Cantidad"].abs()

    df_new = df_new[["Semana", "Almacen", "Producto", "Cantidad"]]

    if True:#df["Semana"].max() < df_new["Semana"].min():

        # Concatenar los DataFrames verticalmente
        df_updated = pd.concat([df, df_new], ignore_index=True)
    else:
        df_updated = df
    return df_updated



def import_data(csv):
    # Leer el archivo CSV subido
    df = pd.read_csv(csv, encoding="latin1", header=0, sep=";")
    # Corregimos el nombre de las columnas:
    df.columns = [
        'Fecha', 'Dia semana', 'Estado doc.', 'Almacen', 'Producto', 'Subfamilia',
        'Nº productos', 'Nº Documentos', 'Nº items (lineas)', 'Cantidad', 'Importe bruto (euro)',
        'Importe dto. (euro)', '% Dto. Efectivo', 'Importe (euro)', 'Margen Efectivo (euro)',
        'Coste total (euro)', '% Margen Efectivo', 'Unnamed: 17']

    # Convertimos la columna 'Fecha' a tipo datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y")
    df['Cantidad'] = df['Cantidad'].str.replace(',', '.')
    df['Cantidad'] = df['Cantidad'].astype('float')

    # Ajustamos las fechas para que el día sea el primer día de la semana (lunes)
    df['Semana'] = df['Fecha'] - pd.to_timedelta(df['Fecha'].dt.dayofweek, unit='D')

    # Pasamos todos los valores a absoluto:
    df["Cantidad"] = df["Cantidad"].abs()

    df = df[["Semana", "Almacen", "Producto", "Cantidad"]]

    return df


def load_data(df):


    try:

        # Asegurarse de que el directorio 'data' existe
        if not os.path.exists('data'):
            os.makedirs('data')

        # Guardar el archivo CSV en el directorio 'data'
        df.to_csv("data/datos_entrenamiento_modelo.csv", index=False)
        print("Archivo guardado en data/datos_entrenamiento_modelo.csv")

        # Usar GitPython para hacer commit y push al repositorio de GitHub
        repo = git.Repo('.')
        repo.git.add('data/datos_entrenamiento_modelo.csv')
        repo.index.commit('Actualizar datos de entrenamiento')

        # Obtener el token de acceso personal desde la variable de entorno
        token = os.getenv('GITHUB_TOKEN')
        if token is None:
            raise ValueError("El token de GitHub no está configurado como variable de entorno.")

        # Configurar la URL remota con el token
        repo_url = f"https://{token}@github.com/arierabr/ausmar_beta.git"  # Reemplaza 'usuario' y 'repo' con tus valores
        origin = repo.remote(name='origin')
        origin.set_url(repo_url)

        # Hacer push al repositorio remoto
        origin.push()
        print("Archivo subido al repositorio de GitHub")

    except Exception as e:
        print(f"Error al guardar el archivo: {e}")


def update_pedidos(csv):
    try:
        # Leer el archivo CSV subido
        pedidos = pd.read_csv(csv, encoding="latin1", header=0, sep=";")

        # Asegurarse de que el directorio 'data' existe
        if not os.path.exists('data'):
            os.makedirs('data')

        # Guardar el archivo CSV en el directorio 'data'
        pedidos.to_csv("data/pedidos.csv", index=False)
        print("Archivo guardado en data/pedidos.csv")

        # Usar GitPython para hacer commit y push al repositorio de GitHub
        repo = git.Repo('.')
        repo.git.add('data/pedidos.csv')
        repo.index.commit('Actualizar pedidos')

        # Obtener el token de acceso personal desde la variable de entorno
        token = os.getenv('GITHUB_TOKEN')
        if token is None:
            raise ValueError("El token de GitHub no está configurado como variable de entorno.")

        # Configurar la URL remota con el token
        repo_url = f"https://{token}@github.com/arierabr/ausmar_beta.git"  # Reemplaza 'usuario' y 'repo' con tus valores
        origin = repo.remote(name='origin')
        origin.set_url(repo_url)

        # Hacer push al repositorio remoto
        origin.push()
        print("Archivo subido al repositorio de GitHub")

    except Exception as e:
        print(f"Error al guardar el archivo: {e}")


def update_stock(csv):
    try:
        # Leer el archivo CSV subido
        inventario = pd.read_csv(csv, encoding="latin1", header=0, sep=";")

        # Asegurarse de que el directorio 'data' existe
        if not os.path.exists('data'):
            os.makedirs('data')

        # Guardar el archivo CSV en el directorio 'data'
        inventario.to_csv("data/inventario.csv", index=False)
        print("Archivo guardado en data/inventario.csv")
        # Usar GitPython para hacer commit y push al repositorio de GitHub
        repo = git.Repo('.')
        repo.git.add('data/inventario.csv')
        repo.index.commit('Actualizar inventario')

        # Obtener el token de acceso personal desde la variable de entorno
        token = os.getenv('GITHUB_TOKEN')
        if token is None:
            raise ValueError("El token de GitHub no está configurado como variable de entorno.")

       # Configurar la URL remota con el token
        repo_url = f"https://{token}@github.com/arierabr/ausmar_beta.git"  # Reemplaza 'usuario' y 'repo' con tus valores
        origin = repo.remote(name='origin')
        origin.set_url(repo_url)

        # Hacer push al repositorio remoto
        origin.push()
        print("Archivo subido al repositorio de GitHub")

    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

def week_number (data_path):
    df = pd.read_csv(data_path)
    df["Semana"] = pd.to_datetime(df["Semana"])
    dt = df["Semana"].max()
    week_number = dt.isocalendar()[1]
    return week_number

