import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("k.csv")
#Category,Message


x = data['Message']
y=data['Category']

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(x, y)
#
# Create an instance of LogisticRegression classifier
#
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
#
# Fit the model
#
lr.fit(X_train, y_train)
#
# Create the predictions
#



for no in range(0,20):

    pred=lr.predict([X_train[no]])

    print(vectorizer.inverse_transform([X_train[no]]))
    print(label_encoder.inverse_transform([y_train[no]]))

    print(pred)

# Use metrics.accuracy_score to measure the score

