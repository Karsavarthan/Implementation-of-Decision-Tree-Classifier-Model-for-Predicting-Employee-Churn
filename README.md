# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Karsavarthan R R
RegisterNumber:  212223230100


import pandas as pd
import numpy as np
df = pd.read_csv('Employee.csv')
df

df.head(5)

df.isnull().sum()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['salary']=le.fit_transform(df["salary"])
df.head()

x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,200,6,0,1,2]])
```

## Output:

Data:

![Screenshot 2025-05-14 230043](https://github.com/user-attachments/assets/80546389-2093-45c0-9e53-288918a6afe0)


Data Head:

![Screenshot 2025-05-14 230056](https://github.com/user-attachments/assets/e26fa10b-e41e-48db-b663-9daf6e2bb746)


Null Dataset:

![Screenshot 2025-05-14 230102](https://github.com/user-attachments/assets/f66bdd6d-820c-4e9b-bc6f-fd21685c111a)


Values count in left column:

![Screenshot 2025-05-14 230109](https://github.com/user-attachments/assets/55de5ae0-08a8-4507-8854-b6eeb66aacb8)


Dataset transformed head:

![Screenshot 2025-05-14 230124](https://github.com/user-attachments/assets/c3e3ef34-97fb-480c-b60f-59902295cda5)


x.head():

![Screenshot 2025-05-14 230138](https://github.com/user-attachments/assets/4b4a2a1f-fd4c-4158-94d8-1a5cd05c5200)


y.head():

![Screenshot 2025-05-14 230150](https://github.com/user-attachments/assets/e2cf5a83-c9f4-4826-a05d-23f7646c4e94)


Accuracy:

![Screenshot 2025-05-14 230200](https://github.com/user-attachments/assets/a6e146e0-bef2-4487-94b0-7e0cd3769d9e)


Data prediction:

![Screenshot 2025-05-14 230213](https://github.com/user-attachments/assets/a397a24f-b843-4b1f-98b9-7573ec094f03)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
