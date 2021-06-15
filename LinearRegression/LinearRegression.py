import numpy as np
from sklearn.linear_model import LinearRegression

#For explanation see documentation of "PolynomialLinearRegression.py"
def lin_reg(x,y):
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)
    print('R^2 is :', r2)
    print('interception (b0) with y axis:', model.intercept_)
    print('slope of the function', model.coef_)