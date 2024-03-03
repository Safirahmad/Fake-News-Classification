import streamlit as st 
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
# # Load data
#loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('train.csv')
# print the first 8 rows of the dataframe
news_dataset.head(8)
# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')


# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

print(news_dataset['content'])

#See values are fill or not 
news_dataset.isnull().sum()

# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

print(X)
print(Y)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)

print(Y)

Y.shape

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

X_new=X_test[40]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

print(Y_test[3])



# # website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vectorizer.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    elif pred ==0:
        st.write('The News Is Real')
    else:
       print("sorry")
data = {'id': [1, 2, 3,4,5,6,7,8,9,10], 'Comments': ['daniel j flynn flynn hillari clinton big woman campu breitbart', 'consortiumnew com truth might get fire', 'jessica purkiss civilian kill singl us airstrik identifi','howard portnoy iranian woman jail fiction unpublish stori woman stone death adulteri','daniel nussbaum jacki mason hollywood would love trump bomb north korea lack tran bathroom exclus video breitbart','life life luxuri elton john favorit shark pictur stare long transcontinent flight','alissa j rubin beno hamon win french socialist parti presidenti nomin new york time','excerpt draft script donald trump q ampa black church pastor new york time','megan twohey scott shane back channel plan ukrain russia courtesi trump associ new york time','aaron klein obama organ action partner soro link indivis disrupt trump agenda'], 'Label':['0','1','0','0','1','0','0','1','1','0']}
df = pd.DataFrame(data)
# Display an interactive DataFrame
st.dataframe(df)
