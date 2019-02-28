# PLOTTING THE RESIDUAL ERRORS GRAPH FOR BOSTON DATASET USING MULTIPLE LINEAR REGRESSION
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics,datasets
boston=datasets.load_boston(return_X_y=False)
X=boston.data
y=boston.target
#splitting training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)
#coefficient matrix
print("The coefficients are:",reg.coef_)
#variance score calculation
print("The varian score is:",reg.score(X_test,y_test))
# PLOTTING THE RESIDUAL ERROR GRAPH
plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train,color="green",s=10,label="Train data")
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,color="red",s=10,label="Test Data")
# zero error line
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)
plt.legend(loc="upper right")
plt.title("Residual Error Graph")
plt.show()
