import numpy as np
import pandas as pd

Data = pd.read_csv(r'heart.csv')

shuffle = Data.sample(frac=1)
training_size = int(0.6 * len(Data))
X_training = shuffle[:training_size]
X_test = shuffle[training_size:]
Y_training = X_training.values[:, -1]
Y_test = X_test.values[:, -1]
Fitting_Result = None
b = 0
alpha = 0.001
lambda_parameter = 0.001
iterators = 1000


def Fitting(X, Y, lr):
    samples, features = X.shape
    X = X.values.astype(float)
    Y_ = np.where(Y <= 0, -1, 1)

    fittingResult = np.zeros((features, 1)).astype(float)

    for k in range(iterators):
        for index, i in enumerate(X):

            condition = Y_[index] * (np.dot(i.reshape(1, 14), fittingResult) - b) >= 1
            if condition:
                fittingResult -= lr * (2 * lambda_parameter * fittingResult)
            else:
                fittingResult -= lr * (2 * lambda_parameter * fittingResult - np.dot(Y_[index], i.reshape(14, 1)))
    return fittingResult


def predict(X, f):
    T = np.dot(X, f) - b
    return  np.sign(T)


def Accuracy(YTest, YPredict):
    R = 0
    F = 0
    for i in range(len(YTest)):
        if YPredict[i] == YTest[i]:
            R += 1
        else:
            F += 1
    Final_Accuracy = (R / len(YTest)) * 100
    return Final_Accuracy


Fitting_Result = Fitting(X_training, Y_training, alpha)

Y_predict = predict(X_test, Fitting_Result)

First_accuracy = Accuracy(Y_test, Y_predict)
print(First_accuracy)
while alpha <= 10:
    fit = Fitting(X_training, Y_training, alpha)
    pred = predict(X_test, fit)
    Printed_Accuracy = Accuracy(Y_test, pred)
    print(Printed_Accuracy)
    alpha *= 10











