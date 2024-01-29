
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
#### Converting Target Variables into Categorical Data type by using Quantile 

```python
print((df.Sales).quantile(0.25))
print((df.Sales).quantile(0.50))
print((df.Sales).quantile(0.75))

Sales_cat = []
for value in df['Sales']:
    if value<=7.49:
        Sales_cat.append("low")
    else:
        Sales_cat.append("high")

df["Sales_cat"]= Sales_cat
```

## [Rndom Forest Classifier Notebook]('https://github.com/vaibhavkatkar3001/Random-Forest-Classifire/blob/main/company_data_prj_9.ipynb')

---

### _Business impact_

So I created a model using `RANDOM FOREST CLASSIFIRE` to target sales. And the 82.5 % of accuracy is tell us that it is a pretty fair fitt the model.


---



##  **2 . Fraud Check project**
### _Problem statement_

Use Random Forest to prepare a model on fraud data 
Treating those who have taxable_income <= 30000 as "Risky" and others are "Good".

---
### _Method_
I used `Random Forest Classifire` to predict risky/good by using taxable income

### This code imports `Random Forest`

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
df['Undergrad_le'] = le.fit_transform(df['Undergrad'])
df['Marital_Status'] = le.fit_transform(df['Marital.Status'])
df['Urban_lr'] = le.fit_transform(df['Urban'])

Taxabale_Income = []
for value in df['Taxable.Income']:
    if value<=30000:
        Taxabale_Income.append("Risky")
    else:
        Taxabale_Income.append("Good")

df["Taxable_Income"]= Taxabale_Income

```
----

## [Random Forest Notebook]('https://github.com/vaibhavkatkar3001/Capstone-Project/blob/main/fraud_check_prj_10.ipynb')

---
### _Business impact_
So, we created a model using `Random Forest` which can predict risky/good by using taxable income. And the 76.66 % of accuracy is tell us that it is a pretty fair fit the model.

---
