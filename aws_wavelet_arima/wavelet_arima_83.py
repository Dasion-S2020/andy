import pywt
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from modwt import modwt, modwtmra
from sklearn import preprocessing
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

df = pd.read_csv('./aapl_indicators_nosplit.csv')
data = df['close']

smooth_order = (20, 1, 0)
detail_order = (1, 1, 0)

wt = modwt(data, 'db2', 6)
c = modwtmra(wt, 'db2')
detail = c[:6]
smooth = c[6]

train_n = smooth.shape[0]*4//5
smooth_train = smooth[:train_n]
smooth_test = smooth[train_n:]
detail_train = []
detail_test = []
for detail_series in detail:
    detail_train.append(detail_series[:train_n])
    detail_test.append(detail_series[train_n:])

def train_arima(train, test, order):

    history = [x for x in train]
    predictions = []

    len_test = len(test)

    for i in range(len_test):

        print(str(i+1) + "/" + str(len_test))
        #wavelet transform on historical data
        #predict next term in smooth series
        try:
            model = ARIMA(history, order=order)
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
        except Exception:
            model = ARIMA(smooth, order=(1,1,0))
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
            
        predictions.append(yhat)
        history.append(test[i])
    return(predictions)

#make predictions
pred_list = []
pred_list.append(np.concatenate(train_arima(smooth_train, smooth_test, smooth_order)))
for i in range(len(detail_train)):
    pred_list.append(np.concatenate(train_arima(detail_train[i], detail_test[i], detail_order)))
pred = pred_list[0]
for i in range(1, len(pred_list)):
    pred += pred_list[i]
actual = data.values[train_n:]

plt.plot(pred, color='r')
plt.plot(actual, color='g')
plt.savefig('./wavelet_arima.png')

#calculate metrics
def mean_absolute_percentage_error(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return(mape)

mse = mean_squared_error(actual, pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(actual, pred)
mape = mean_absolute_percentage_error(actual, pred)

print("mse, rmse, mae, mape:" + str((mse,rmse,mae,mape)))


