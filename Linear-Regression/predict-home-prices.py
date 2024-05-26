# linear regression single variable
# linear equation y = mx + c
# price = m * area +  b


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
print(df)

# %matplotlib inline
plt.scatter(df.area, df.price, color='red', marker='+')
plt.xlabel("Area(sqf)")
plt.ylabel("Price(US$)")
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
