# Using KNN Algorithm to predict if a person will have diabetes or not
# importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# loading the dataset

data = pd.read_csv('diabetes.csv')

print(data.head())

# Replace columns like [Gluscose,BloodPressure,SkinThickness,BMI,Insulin] with Zero as values with mean of respective column


zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
# for col in zero_not_accepted:
#     for i in data[col]:
#         if i==0:
#             colSum = sum(data[col])
#             meanCol=colSum/len(data[col])
#             data[col]=meanCol

for col in zero_not_accepted:
    data[col]= data[col].replace(0,np.NaN)
    mean = int(data[col].mean(skipna=True))
    data[col] = data[col].replace(np.NaN,mean)

# extracting independent variables

X = data.iloc[:,0:8]
# extracting dependent variable

y = data.iloc[:,8]
# Explorning data to know relation before processing

sns.heatmap(data.corr())

plt.figure(figsize=(25,7))
sns.countplot(x='Age',hue='Outcome',data=data,palette='Set1')

# splitting dataset into training and testing set


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# feature scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# loading model - KNN

classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
# fitting model

classifier.fit(X_train,y_train)

# making predictions

y_pred = classifier.predict(X_test)
# evaluating model

conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print(f1_score(y_test,y_pred))

# accuracy

print(accuracy_score(y_test,y_pred))

plt.show()