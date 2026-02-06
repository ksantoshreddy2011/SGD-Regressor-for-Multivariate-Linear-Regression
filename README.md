# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the weight and bias.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SANTHOSH REDDY K
RegisterNumber: 212225240137 
*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
X = np.array([
    [1,2],[2,1],[3,4],[4,3],[5,5],
    [6,7],[7,6],[8,9],[9,8],[10,10],
    [2,3],[3,2],[4,5],[5,4],[6,6],
    [7,8],[8,7],[9,10],[10,9],[11,11]
])
y = np.array([
    5,6,10,11,15,
    19,20,26,27,30,
    8,9,14,15,18,
    23,24,29,30,33
])

model = SGDRegressor(max_iter=2000, eta0=0.01, learning_rate='constant', random_state=42)

model.fit(X, y)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

y_pred = model.predict(X)

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

plt.show()

```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house] <img width="1414" height="758" alt="EXP4OUTPUT" src="https://github.com/user-attachments/assets/9d764ef8-8885-4f92-bcf9-bdd7998816cf" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
