#Linear regression
# - Write a function to compute hypothesis
# - Write a function to compute the cost
# - Write a function to compute the derivative
# - Write update parameters logic in the main function

import numpy as np
import pandas as pd


def comp_hx(X):
    hx=[0]*len(X[0])
    theta = [0]*len(X[0])
    new_hx=[]
    for i in range(len(hx)):
        hx_array = []
        hx_value=0
        for sample in range(0,len(X)):
            hx_value += theta[0]*1 + np.dot(theta, X[sample])
            hx_array.append(hx_value)
        new_hx.append(sum(hx_array))
    return np.array(new_hx)


# compute cost
def compute_cost(hx,y):
    # J = 1/2((hx - y)**2 + ...)
    TSE  = 0
    for i,j in zip(hx,y):
        TSE += (i - j)**2
    j = TSE/2
    return j


def comp_update_theta(hx,X, y, alpha=0.01):
    #Update theta
    theta = [0]*len(X[0])
    print('length', len(theta), len(X[0]))
    new_theta=[]
    for i in range(0,len(X[0])):
        dj_dt= 0
        for j in range(0, len(X[i])):
            X_j = X[i][j]
            y_i = y[i]
            dj_dt += (hx[i] - y_i)*X_j
        new_th_val = theta[i] - (alpha*dj_dt)
        new_theta.append(new_th_val)
    return new_theta



def main():
    # X = [[1, 1, 2],
    #      [2, 3, 4],
    #      [3, 4, 7],
    #      [6,5,7]]
    # y = [2, 3, 5, 6]


    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    df = pd.DataFrame(df)
    features = ['age', 'BMI', 'BP', 'Gender', 'blood_sugar']

    X = df[features].values
    print(type(X))

    target = ['disease_score_fluct']
    y = df[target].values

    hx = comp_hx(X)
    print(hx)
    J = compute_cost(hx,y)
    print(J)

    theta = comp_update_theta(hx,X, y)
    print(hx, "\n\n",J, "\n\n", theta)

if __name__=="__main__":
    main()

