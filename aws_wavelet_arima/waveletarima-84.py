import pywt
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from modwt import modwt, modwtmra
from sklearn import preprocessing
import pmdarima
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

df = pd.read_csv('/Users/andyliu/develop/andy/aapl_indicators_yahoo.csv')
data = df['Close']

train_n = data.shape[0]*4//5
train = np.array(data[:train_n])
test = np.array(data[train_n:])

wt = modwt(data, 'db2', 6)
c = modwtmra(wt, 'db2')
detail = c[:6]
smooth = c[6]

smooth_test = smooth[train_n:]
smooth_train = smooth[:train_n]
detail_train = []
detail_test = []
for i in range(len(detail)):
    detail_train.append(detail[i][:train_n])
    detail_test.append(detail[i][train_n:])

def num_diffs(y_train):
    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    return(n_diffs)

smooth_auto = pmdarima.auto_arima(smooth_train, d=num_diffs(smooth_train), seasonal=False, stepwise=True, suppress_warnings=True, max_p=20, trace=2, error_action='ignore')
detail_models = []
for detail_series in detail_train:
    detail_auto = pmdarima.auto_arima(detail_series, d=num_diffs(detail_series), seasonal=False, stepwise=True, suppress_warnings=True, max_p=6, trace=2, error_action='ignore')
    detail_models.append(detail_auto)

def arima_train(train, test, arima, max_p):

    tot = len(test)
    history = [x for x in train]
    predictions = []
    for t in range(tot):
        try:
            print(str(t+1) + "/" + str(tot))
            arima.fit(history)
            yhat = arima.predict()[0]
            predictions.append(yhat)
            history.append(test[t])
            
        except Exception:
            print("Recalculating ARIMA Parameters...")
            arima = pmdarima.auto_arima(history, d = num_diffs(history), seasonal=False, stepwise=True, suppress_warnings=True, max_p=max_p, trace=2, error_action='ignore')
            arima.fit(history)
            yhat = arima.predict()[0]
            predictions.append(yhat)
            history.append(test[t])
            
    return(predictions)

smooth_pred = arima_train(smooth_train, smooth_test, smooth_auto, 15)
detail_pred = []
for i in range(len(detail_models)):
    d = arima_train(detail_train[i], detail_test[i], detail_models[i], 5)
    detail_pred.append(d)

pred = np.array(smooth_pred)
for i in detail_pred:
    pred += np.array(i)

np.save('./pred.npy', pred)

actual = test

plt.plot(pred, color='r')
plt.plot(actual, color='g')
plt.savefig('./arima84.png')

def mean_absolute_percentage_error(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return(mape)

mse = mean_squared_error(actual, pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(actual, pred)
mape = mean_absolute_percentage_error(actual, pred)
print("mse, rmse, mae, mape:" + str((mse,rmse,mae,mape)))