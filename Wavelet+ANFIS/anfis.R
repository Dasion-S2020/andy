#read data
setwd("/Users/andyliu/develop/andy/Wavelet+ANFIS")
train_x <- read.csv(file = 'train_x.csv')
test_x <- read.csv(file = 'test_x.csv')
train_y <- read.csv(file = 'train_y.csv')
test_y <- read.csv(file = 'test_y.csv')
train <- cbind(train_x, test_x)

