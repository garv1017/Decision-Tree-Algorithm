#!/usr/bin/env python
# coding: utf-8

# # GRIP - The Spark Foundation Internship Program

# ## Data Science and Business Analytics

# ## Prediction using Decision Tree Algorithm

# ## Author : Alladi Varsha

# ## Problem Statement : 

# In[ ]:


To predict right class accordingly, if we feed new class to classifier.


# In[ ]:


In order to predict the right class we use an algorithm called "Decision Tree Algorithm."


# In[ ]:


Decision Tree Algorithm : It is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred solving Classification Problems.


# ## Code : 

# ## Importing all Necessary Libraries 

# In[1]:


#importing all libraries relevent to that
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[2]:


#ignore warnings
import warnings as wg
wg.filterwarnings("ignore")


# ## Loading The Data

# In[13]:


#loading the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
iris_df = pd.DataFrame(np.c_[iris['data'],iris['target']],
             columns = np.append(iris['feature_names'], ['target']))


# ## Data Manipulation 

# In[5]:


iris_df.head() 


# In[6]:


iris_df.tail()


# In[7]:


iris_df.info()


# In[8]:


iris_df.describe()


# In[9]:


iris_df.isnull().sum()


# ## Data Visualisation

# In[10]:


sns.pairplot(iris_df, hue = 'target') 


# In[14]:


# Count the target class
sns.countplot(iris_df['target'])


# In[15]:


# heatmap of DataFrame
plt.figure(figsize=(16,9))
sns.heatmap(iris_df)


# In[16]:


iris_df.corr()


# In[17]:


# Heatmap of Correlation m# Heatmap of Correlation matrix of iris DataFrame
plt.figure(figsize=(20,20))
sns.heatmap(iris_df.corr(), annot = True, cmap ='coolwarm', linewidths=2) 


# In[18]:


iris_df2 = iris_df.drop(['target'], axis = 1)
print("The shape of 'iris_df2' is : ", iris_df2.shape)


# In[22]:


iris_df2.corrwith(iris_df.target).index


# ## Split DataFrame

# In[23]:


# input variable
X = iris_df.drop(['target'], axis = 1) 
X.head()


# In[24]:


# output variable
y = iris_df['target']
y.head(6)


# In[25]:


# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)


# In[26]:


X_train


# ## Machine Learning Model Building

# In[28]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ## Feature Scaling

# In[29]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# ## Decision Tree Classifier

# In[30]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)


# In[31]:


# Train with Standard scaled Data
dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


# ## Fit the Classifier

# In[32]:


clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)


# In[33]:


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)


# In[ ]:


In this way, We can successfully create Decision Tree Classifier and visualise it graphically.

