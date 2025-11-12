# Car
Developed a supervised machine learning model capable of predicting car prices using Linear Regression.

## SUPERVISED LEARNING: REGRESSION ANALYSIS

-----

### Step #1 Importing the required libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn
import sklearn

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
```

-----

### Step #2 Loading the dataset

```python
# TODO: Get the datset ./AI_Invasion_In-Class_Dataset.xlsx form your AI Invasion
# Study Pack
# Note: You can use pandas read_excel to read file with xlsx format

df = pd.read_excel("data/AI_Invasion_In-Class_Dataset.xlsx")

df.head()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4487 entries, 0 to 4486
Data columns (total 8 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Location            4487 non-null   object 
 1   Maker               4487 non-null   object 
 2   Model               4487 non-null   object 
 3   Year                4487 non-null   int64  
 4   Colour              4487 non-null   object 
 5   Amount (Million ₦)  4487 non-null   float64
 6   Type                4487 non-null   object 
 7   Distance_Km         2932 non-null   float64
dtypes: float64(2), int64(1), object(5)
memory usage: 280.6+ KB
```

```python
df.describe()
```

-----

### Step #3 Clean the dataset

```python
df.columns
```

```
Index(['Location', 'Maker', 'Model', 'Year', 'Colour', 'Amount (Million ₦)',
       'Type', 'Distance_Km'],
      dtype='object')
```

```python
# Check for missing value
df.isnull().sum()
```

```
Location                 0
Maker                    0
Model                    0
Year                     0
Colour                   0
Amount (Million ₦)       0
Type                     0
Distance_Km           1555
dtype: int64
```

```python
# fill up missing values in Distance_Km will the mean
mean_value = df["Distance_Km"].mean()
print(mean_value)

df["Distance_Km"].fillna(mean_value, inplace=True)
```

```
101038.32128240108
```

```python
# Check and make sure all missing valuen have been filled
df.isnull().sum()
```

```
Location              0
Maker                 0
Model                 0
Year                  0
Colour                0
Amount (Million ₦)    0
Type                  0
Distance_Km           0
dtype: int64
```

```python
# The main of this section is to rename the different
# class in our categorigal feature that were not properly named.
# or chanage the data type of a column

cat_features = {
 "Location",
 "Model",
 "Maker",
 "Year",
 "Colour",
 "Type",
}

for cat_feature in cat_features:
 print(cat_feature, df[cat_feature].unique(), sep=":")
 print("#"*50)
```

```python
# Drop the Model feature
df.drop("Model", axis=1, inplace=True)
df.head()
```

```python
cat_features = {
 "Location",
 "Maker",
 "Year",
 "Colour",
 "Type",
}
# Using cat.codes for Label Encoding

for cat_feature in cat_features:
 df[f"{cat_feature}_cat"] = df[cat_feature].astype('category')
 df[f"{cat_feature}_cat"] = df[f"{cat_feature}_cat"].cat.codes


# Read more on Pandas get_dummies

df.head()
```

-----

### Step #4 Feature Selection

```python
X = df[[
    "Distance_Km",
    "Location_cat",
    "Maker_cat",
    "Year_cat",
    "Colour_cat",
    "Type_cat"
]]

y = df["Amount (Million ₦)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.head()
```

-----

### Step #5 Training the Model

#### Linear Regression

##### Inserting your data into the Linear Regression model i.e Train your model

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
```

```
LinearRegression()
```

-----

## Step #6 Make predictions

```python
y_pred = reg.predict(X_test)
y_pred
```

```
array([16.47919229, 22.68792222, -1.19896663, ...,
       12.39070908,  5.7891761 , 12.38137356])
```

-----

## Step #7 Check the accuracy of the model

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE", mean_absolute_error(y_test,y_pred))
print("MSE", mean_squared_error(y_test,y_pred))
print("R2", r2_score(y_test,y_pred))
```

```
MAE 7.423719047911674
MSE 1461.3551579893542
R2 -0.1630113883072049
```

-----

## Support Vector Regression (Activity)

```python
from sklearn.svm import SVR

sv_reg = SVR()
sv_reg.fit(X_train, y_train)
y_pred = sv_reg.predict(X_test)
print("MAE",mean_absolute_error(y_test,y_pred))
```

```
MAE 7.9238553867455375
```

-----

## Random Forest (Activity)

```python
# Code to implement Random Forest will go here
```

```python
# Empty code block
```

-----

## 👨🏽‍💻 Author

Japhet Ujile
📧 [assistant.rawlings@gmail.com](mailto:assistant.rawlings@gmail.com)
🌐 [GitHub](https://github.com/assistantrawlings-lgtm) | [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/japhet-ujile-838442148%3Futm_source%3Dshare%26utm_campaign%3Dshare_via%26utm_content%3Dprofile%26utm_medium%3Dandroid_app%5D)
