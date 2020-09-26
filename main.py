import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import sklearn as sk


yelp = pd.read_csv('/Users/Magic Systems/Desktop/yelp.csv')


print(yelp.head())
print(yelp.info())
print(yelp.describe())


yelp['text length'] = yelp['text'].apply(len)
print( yelp['text length'] )

# # EDA

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
plt.show()

sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
plt.show()

# count plot
sns.countplot(x='stars',data=yelp,palette='rainbow')
plt.show()

#Use groupby to get the mean values of the numerical columns
stars = yelp.groupby('stars').mean()
print(stars)


#Use the corr() method on that groupby dataframe to produce this dataframe:
print(stars.corr())


#  use seaborn to create a heatmap based off that .corr() dataframe
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
plt.show()

# NLP Classification Task
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

X = yelp_class['text']
y = yelp_class['stars']


#Import CountVectorizer and create a CountVectorizer object.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


#Use the fit_transform method on the CountVectorizer object and pass in X .
X = cv.fit_transform(X)


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# Training a Model
#  Import MultinomialNB and create an instance of the estimator and call is nb

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)


# Predictions and Evaluations
# Use the predict method off of nb to predict labels from X_test.
predictions = nb.predict(X_test)


# Create a confusion matrix and classification report using these predictions and y_test
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

#  Import TfidfTransformer from sklearn.
from sklearn.feature_extraction.text import  TfidfTransformer
# Import Pipeline from sklearn.
from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Redo the train test split on the yelp_class object.

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels**


pipeline.fit(X_train,y_train)


#Predictions and Evaluation
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


