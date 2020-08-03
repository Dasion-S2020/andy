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

train_n = data.shape[0]*4//5
train = np.array(data[:train_n])
test = np.array(data[train_n:])
test_n = data.shape[0]//5

smooth_order = (5, 1, 1)
detail_order = (1, 1, 0)

history = [x for x in train]
predictions = []
len_test = len(test)

for i in range(len_test):

    print(str(i+1) + "/" + str(len_test))
    #wavelet transform on historical data
    wt = modwt(history, 'db2', 6)
    c = modwtmra(wt, 'db2')
    detail = c[:6]
    smooth = c[6]

    #predict next term in smooth series
    model = ARIMA(smooth, order=smooth_order)
    model_fit = model.fit(disp=0)
    yhat_smooth = model_fit.forecast()[0]

    #predict next term in each detail series
    detail = []
    for detail_series in detail:
        detail_model = ARIMA(detail_series, order=detail_order)
        detail_model_fit = detail_model.fit(disp=0)
        yhat_detail = detail_model_fit.forecast()[0]
        detail.append(yhat_detail)

    yhat = yhat_smooth
    for d in detail:
        yhat += d

    predictions.append(yhat)
    history.append(test[i])

#make predictions
pred = predictions
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


