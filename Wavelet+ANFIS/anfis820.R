setwd("~/develop/andy/Wavelet+ANFIS")
library(frbs)
train_x = read.csv('trainx.csv')
test_x = read.csv('testx.csv')
train_y = read.csv('trainy.csv')
test_y = read.csv('testy.csv')
train_x = train_x[,-1]
test_x = test_x[,-1]
train_y = train_y[,-1]
test_y = test_y[,-1]
train = cbind(train_x, train_y)
