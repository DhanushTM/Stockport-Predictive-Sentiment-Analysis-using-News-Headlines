#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

# Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import metrics
import itertools

import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras

# For Natural Language Processing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('Combined_News_DJIA.csv', encoding = "ISO-8859-1")# if there are utf-8 characters


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df


# In[7]:


train =  df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[8]:


#removing punctuation marks
data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)


# In[9]:


# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)


# In[10]:


' '.join(str(x) for x in data.iloc[1,0:25])


# In[11]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[12]:


headlines[0]


# In[13]:


train['headlines'] = headlines
train_headlines = headlines


# In[14]:


selected_categories = ['headlines']


# # WORD EMBEDDING - Using Count-Vectorizer

# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[15]:


# Implement BAG OF WORDS
countvector = CountVectorizer(ngram_range=(2,2))
train_dataset_cv = countvector.fit_transform(headlines)


# # Random Forest

# In[16]:


# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(train_dataset_cv,train['Label'])


# # Multinomial Naive Bayes

# In[17]:


# implement Multinomial Naive Bayes
nb=MultinomialNB()
nb.fit(train_dataset_cv,train['Label'])


# # Logistic Regression

# In[18]:


# implement Logestic Regression
lr=LogisticRegression()
lr.fit(train_dataset_cv,train['Label'])


# # Support Vector Classifier

# In[19]:


# implement Support vector classifier
svm = SVC(kernel='linear', random_state=0)  
svm.fit(train_dataset_cv,train['Label'])


# # K-Nearest Neighbor

# In[20]:



# implement K-Nearest Neighbor
knn=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
knn.fit(train_dataset_cv,train['Label'])


# # Evaluation

# In[21]:


## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_headlines = test_transform    
test_dataset_cv = countvector.transform(test_transform)


# In[22]:


# Prediciton RandomForest Classifier
predictions_rf_cv = randomclassifier.predict(test_dataset_cv)


# In[23]:


predictions_rf_cv


# In[25]:


# Prediciton Multinomial Naive Bayes
predictions_nb_cv = nb.predict(test_dataset_cv)


# In[26]:


# Prediciton Logestic Regression
predictions_lr_cv = lr.predict(test_dataset_cv)


# In[27]:


# Prediciton K-Nearest Neighbor
predictions_knn_cv = knn.predict(test_dataset_cv)


# In[28]:


# Prediciton Support vector classifier
predictions_svm_cv = svm.predict(test_dataset_cv)


# In[29]:


## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[30]:


# CV-RF
matrix_rf_cv=confusion_matrix(test['Label'],predictions_rf_cv)
print(matrix_rf_cv)
score_rf_cv=accuracy_score(test['Label'],predictions_rf_cv)
print(score_rf_cv)
report_rf_cv=classification_report(test['Label'],predictions_rf_cv)
print(report_rf_cv)


# In[31]:


# CV-NB
matrix_nb_cv=confusion_matrix(test['Label'],predictions_nb_cv)
print(matrix_nb_cv)
score_nb_cv=accuracy_score(test['Label'],predictions_nb_cv)
print(score_nb_cv)
report_nb_cv=classification_report(test['Label'],predictions_nb_cv)
print(report_nb_cv)


# In[32]:


# CV-LR
matrix_lr_cv=confusion_matrix(test['Label'],predictions_lr_cv)
print(matrix_lr_cv)
score_lr_cv=accuracy_score(test['Label'],predictions_lr_cv)
print(score_lr_cv)
report_lr_cv=classification_report(test['Label'],predictions_lr_cv)
print(report_lr_cv)


# In[33]:


# CV-KNN
matrix_knn_cv=confusion_matrix(test['Label'],predictions_knn_cv)
print(matrix_knn_cv)
score_knn_cv=accuracy_score(test['Label'],predictions_knn_cv)
print(score_knn_cv)
report_knn_cv=classification_report(test['Label'],predictions_knn_cv)
print(report_knn_cv)


# In[34]:


# CV-SVM
matrix_svm_cv=confusion_matrix(test['Label'],predictions_svm_cv)
print(matrix_svm_cv)
score_svm_cv=accuracy_score(test['Label'],predictions_svm_cv)
print(score_svm_cv)
report_svm_cv=classification_report(test['Label'],predictions_svm_cv)
print(report_svm_cv)


# In[35]:


# Creating corpus of train dataset
ps = PorterStemmer()
train_corpus = []

for i in range(0, len(train_headlines)):
  
  # Tokenizing the news-title by words
  words = train_headlines[i].split()

  # Removing the stopwords
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in words]

  # Joining the stemmed words
  headline = ' '.join(words)

  # Building a corpus of news-title
  train_corpus.append(headline)


# In[36]:



# Creating corpus of test dataset
test_corpus = []

for i in range(0, len(test_headlines)):
  
  # Tokenizing the news-title by words
  words = test_headlines[i].split()

  # Removing the stopwords
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in words]

  # Joining the stemmed words
  headline = ' '.join(words)

  # Building a corpus of news-title
  test_corpus.append(headline)
     


# In[37]:


down_words = []
for i in list(train['Label'][train['Label']==0].index):
  down_words.append(train_corpus[i])

up_words = []
for i in list(train['Label'][train['Label']==1].index):
  up_words.append(train_corpus[i])
     


# In[47]:


get_ipython().system('pip install wordcloud')


# In[38]:


# Creating wordcloud for down_words
from wordcloud import WordCloud
wordcloud1 = WordCloud(background_color='black', width=3000, height=2500).generate(down_words[1])
plt.figure(figsize=(8,8))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title("Indicating fall in Stock ")
plt.show()


# In[41]:


# Creating wordcloud for up_words
wordcloud2 = WordCloud(background_color='black', width=3000, height=2500).generate(up_words[5])
plt.figure(figsize=(8,8))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title("Indicating rise in Stock ")
plt.show()


# In[50]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[51]:


# Plot Confusion_Matrix RandomForest Classifier
plot_confusion_matrix(matrix_rf_cv, classes=['Down', 'Up'])


# In[52]:


# Plot Confusion_Matrix Multinomial Naive Bayes
plot_confusion_matrix(matrix_nb_cv, classes=['Down', 'Up'])


# In[53]:


# Plot Confusion_Matrix Logestic Regression
plot_confusion_matrix(matrix_lr_cv, classes=['Down', 'Up'])


# In[54]:


# Plot Confusion_Matrix K-Nearest Neighbor
plot_confusion_matrix(matrix_knn_cv, classes=['Down', 'Up'])


# In[55]:


# Plot Confusion_Matrix Support vector classifier
plot_confusion_matrix(matrix_svm_cv, classes=['Down', 'Up'])


# In[ ]:




