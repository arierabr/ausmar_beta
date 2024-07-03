import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pickle

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


def update_models(lista_referencias, df):
    for i in range(len(lista_referencias)):
        referencia = df[df["Producto"] == lista_referencias[i]].groupby("Semana")["Cantidad"].sum().reset_index()
        referencia.set_index(['Semana'], inplace=True)
        referencia.index = pd.to_datetime(referencia.index)
        hw_mul = ets.ExponentialSmoothing(consumos_CA140180_train, trend='mul', damped_trend=False,
                                          seasonal='mul').fit()
        with open(f'modelos/hw_mul_model_{lista_referencias[i]}.pkl', 'wb') as file:
            pickle.dump(hw_mul, file)


def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

