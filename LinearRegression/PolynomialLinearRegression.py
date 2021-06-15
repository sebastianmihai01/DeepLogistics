import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#ValDegree = polynomial degree
#bias = boolean true/false for bias approximations

def populateBy_observation_points(list_x,list_y):
    nr_obs_points = np.size(list_x) #nr of observation points

    #lines for x and y (means)

    meanX = np.mean(list_x)
    meanY = np.mean(list_y)

    #difference between plot and obs point = deviation
    #calculation of the deviation is the following:

    SS_xy = np.sum(list_y * list_x) - nr_obs_points * meanY * meanX
    SS_xx = np.sum(list_x * list_x) - nr_obs_points * meanX * meanX

    #slope:
    slope = SS_xx

    #intersection with y axis:
    intersection = meanY - slope*meanX


def pol_reg(x,y,valDegree,bias):
    try:
        #numpy arrays, specific for lin regression
        x = np.array(x)
        y = np.array(y)
        
        formattedX = PolynomialFeatures(degree=valDegree, include_bias=bias).fit_transform(x)
        model = LinearRegression().fit(formattedX, y)
        score = model.score(formattedX, y)
        intercept, coefficients = model.intercept_, model.coef_
        predictedY = model.predict(formattedX)

        print('R^2: coeff. of determination:', score)
        print('b0: interception with the y axis:', intercept)
        print('coeff:', coefficients, sep='\n')
        print('prediction:', predictedY, sep='\n')
    except Exception as exc:
        print("Errors in any of the arrays 'x' or 'y'")


def plot(list_x, list_y, b):

    #plotting the points
    plt.scatter(list_x, list_y, color="m",marker="X", s=10)
    y_pred = b[0] + b[1] * list_x

    #plotting the line
    plt.plot(list_x, y_pred, color="b")

    plt.xlabel('oX')
    plt.ylabel('oY')
    plt.show()

    coef = populateBy_observation_points(list_x, list_y)
    b_0 = coef[0]
    b_1 = coef[1]

    plot(list_x,list_y,coef)

    #perhaps not working
    #perhaps2 only for polynomials of degree 2
    print("linear regression function: f(x) = ",b_0 ," + ", b_1, "x")