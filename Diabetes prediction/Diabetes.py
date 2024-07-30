import numpy as np
import pandas as py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#loading the diabetes dataset to pandas DataFrame
diabetes_dataset=py.read_csv('diabetes.csv')
py.read_csv
diabetes_dataset.head()
diabetes_dataset.shape
#getting the statistical measures of data
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
#separating the data and labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
print(X)
print(Y)
#Data Standardization
scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
print(standardized_data)
X=standardized_data
Y=diabetes_dataset['Outcome']
print(X)
print(Y)
#Train Test Split
X_train, X_test, Y_train ,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
#Training the model
classifier=svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train,Y_train)
#accuracy score on training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
#accuracy score on test data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
#making predictive system
input_data=(4,110,92,0,0,37.6,0.191,30)

#change the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction=classifier.predict(std_data)
print(prediction)

if(prediction[0]==1):
        print("The person is Diabetic")
else:
        print("The person is not Diabetic")

