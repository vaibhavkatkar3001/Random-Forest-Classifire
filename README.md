
# **Random Forest Projects**

This project is about 'RANDOM FOREST' Classification.
In this, I have completed two projects.
- Company data project
* Fraud Check project
---
##  **1 . Company project**
### _Problem statement_

A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
Approach - A Random Forest can be built with target variable Sales (we will first convert it in categorical variable) & all other variable will be independent in the analysis. 

### _Method_
I used `RANDOM FOREST Classifire` to target variable sales.

### This code imports `Random Forest Classifire`

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
```
---
#### Encoding 

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['ShelveLoc_le'] = le.fit_transform(df.ShelveLoc)
df['Urban_le'] = le.fit_transform(df.Urban)
df['US_le'] = le.fit_transform(df.US)
```
----

## [Rndom Forest Classifier Notebook]('')

---

### _Business impact_

So I created a model using KNN which can predict wheather person can purchase Iphone or not. And the 86 % of accuracy is tell us that it is a pretty fair fitt the model.


---



##  **2 . Houseprice project**
### _Problem statement_

To Predict the Price of Bangalore House.

---
### _Method_
I used `KNN Regressor` to predict price of Bangalore house

### This code imports `Regressor`

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,y_train)
y_pred = model_fit.predict(X_test)
results = r2_score(y_test,y_pred)
print(results) 
``` 
---

## [KNN Regressor Notebook]('https://github.com/vaibhavkatkar3001/Capstone-Project/blob/main/houseprice_prj6.ipynb')

---
### _Business impact_
So, we created a model using KNN which can Predict the Price of Bangalore House. And the 96 % of accuracy is tell us that it is a pretty fair fit the model.

---
