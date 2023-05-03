#!/usr/bin/env python
# coding: utf-8

# # Healthcare - Persistency of a drug
# #### One of the challenge for all Pharmaceutical companies is to understand the persistency of drug as per the physician prescription. To solve this problem ABC pharma company approached an analytics company to automate this process of identification.

# In[ ]:





# In[1]:


import pandas as pd   # For data manupulation using dataframes
import numpy as np    # For Statistical Analysis
import math


# In[2]:


d=pd.read_excel('Healthcare_dataset.xlsx')
d.head()


# In[3]:


# no of rows and columns
d.shape


# In[4]:


# Datatypes of columns and non-null values
d.info()


# In[5]:


# Function to identify numeric features
def numeric_features(dataset):
    numeric_col = dataset.select_dtypes(include=['number']).columns
    return numeric_col

# Function to identify categorical features
def categorical_features(dataset):
    categorical_col = dataset.select_dtypes(exclude=['number']).columns
    return categorical_col


# In[6]:


# display numeric and categorical features
def display_numeric_categoric_feature(dataset):
    numeric_columns = numeric_features(dataset)
    print("Numeric Features:")
    print(numeric_columns)
    print("===="*20)
    categorical_columns = categorical_features(dataset)
    print("Categorical Features:")
    print(categorical_columns)


# In[7]:


display_numeric_categoric_feature(d)


# In[8]:


# total null values in the dataset
d.isnull().sum()


# In[9]:


d.isna().apply(pd.value_counts)


# There is no null value in the dataset
# 

# In[10]:


# Description of numerical columns
d.describe()


# In[11]:


# list of columns 
d.columns


# # Exploratory Data analysis

# In[12]:


# Remove duplicate rows
d=d.drop_duplicates()
d


# There is no duplicate row

# In[13]:


#Drop ptid column
d.drop(columns='Ptid',axis=1,inplace=True)


# # Visualization and Preprocessing

# In[14]:


import matplotlib.pyplot as plt       # For Data Visualisation
import seaborn as sns                 # for statistical Data Visualisation
import warnings
warnings.filterwarnings('ignore')


# ## Univariate Analysis for Continuous Columns

# In[15]:


d.hist()


# There is uneven distribution of data in both continuous columns

# In[16]:


d.boxplot()


# There are outliers in both continuous columns

# In[17]:


sns.pairplot(d) #visuals representation of Correlation between all continuous columns.


# There is very low correlation between both the numerical columns

# ## Univariate Analysis for Categorical Columns

# In[20]:


plt.figure(figsize=(15,3))
plt.subplot(1,4,1);d['Gender'].value_counts().plot(kind='bar',color=['C0','C1']);plt.title('Gender')
plt.subplot(1,4,2);d['Race'].value_counts().plot(kind='bar',color=['C3','C4']);plt.title('Race')
plt.subplot(1,4,3);d['Ethnicity'].value_counts().plot(kind='bar',color=['C0','C1']);plt.title('Ethnicity')
plt.subplot(1,4,4);d['Region'].value_counts().plot(kind='bar',color=['C3','C4']);plt.title('Region')


# In[24]:


plt.figure(figsize=(15,3))
plt.subplot(1,3,1);d['Age_Bucket'].value_counts().plot(kind='bar',color=['C0','C1']);plt.title('Age_Bucket')
plt.subplot(1,3,2);d['Ntm_Specialist_Flag'].value_counts().plot(kind='bar',color=['C0','C1']);plt.title('Ntm_Specialist_Flag')
plt.subplot(1,3,3);d['Ntm_Speciality_Bucket'].value_counts().plot(kind='bar',color=['C3','C4']);plt.title('Ntm_Speciality_Bucket')


# In[25]:


plt.figure(figsize=(25,3))
plt.subplot(1,2,2);d['Ntm_Speciality'].value_counts().plot(kind='bar',color=['C3','C4']);plt.title('Ntm_Speciality')


# In[26]:


plt.figure(figsize=(15,3))
plt.subplot(1,3,1);d['Gluco_Record_Prior_Ntm'].value_counts().plot(kind='bar',color=['C0','C1']);plt.title('Gluco_Record_Prior_Ntm')
plt.subplot(1,3,2);d['Gluco_Record_During_Rx'].value_counts().plot(kind='bar',color=['C3','C4']);plt.title('Gluco_Record_During_Rx')
plt.subplot(1,3,3);d['Dexa_During_Rx'].value_counts().plot(kind='bar',color=['C0','C1']);plt.title('Dexa_During_Rx')


# In[27]:


plt.figure(figsize=(15,3))
plt.subplot(1,3,1);d['Frag_Frac_Prior_Ntm'].value_counts().plot(kind='bar',color=['C3','C4']);plt.title('Frag_Frac_Prior_Ntm')
plt.subplot(1,3,2);d['Frag_Frac_During_Rx'].value_counts().plot(kind='bar',color=['C0','C1']);plt.title('Frag_Frac_During_Rx')
plt.subplot(1,3,3);d['Risk_Segment_Prior_Ntm'].value_counts().plot(kind='bar',color=['C3','C4']);plt.title('Risk_Segment_Prior_Ntm')


# In[ ]:





# In[ ]:





# In[ ]:





# ## Bivariate Analysis

# In[28]:


sns.catplot(
    data=d, y="Persistency_Flag", hue="Gender", kind="count",
    palette="pastel", edgecolor=".6",)


# In[29]:


sns.catplot(
    data=d, y="Persistency_Flag", hue="Race", kind="count",
    palette="pastel", edgecolor=".6",)


# In[30]:


sns.catplot(
    data=d, y="Persistency_Flag", hue="Ethnicity", kind="count",
    palette="pastel", edgecolor=".6",)


# In[31]:


sns.catplot(
    data=d, y="Persistency_Flag", hue="Region", kind="count",
    palette="pastel", edgecolor=".6",)


# In[32]:


sns.catplot(
    data=d, y="Persistency_Flag", hue="Age_Bucket", kind="count",
    palette="pastel", edgecolor=".6",)


# Through bivariate analysis did not get significant insights

# ## Categorical Feature Selection using  Chi - Square test of Independence

# In[33]:


cat_col=d.drop(columns=['Dexa_Freq_During_Rx', 'Count_Of_Risks'])
cat_col.head()


# In[34]:


categorical_columns = categorical_features(d)


# In[35]:


# Convert object to category
d[categorical_columns]=d[categorical_columns].astype("category")


# In[36]:


d.info()


# In[37]:


# encoding categorical features into numeric
d[categorical_columns]=d[categorical_columns].apply(lambda x: x.cat.codes)


# In[38]:


d.info()


# In[39]:


x=d[categorical_columns].drop(columns=['Persistency_Flag'])
y=d['Persistency_Flag']


# ## Categorical Feature Selection using sklearn library and chi2 and SelectKbest function

# In[40]:


from sklearn.feature_selection import chi2, SelectKBest


# In[41]:


cs= SelectKBest (score_func = chi2, k= "all")
cs.fit(x,y)
feature_score = pd.DataFrame({"Score":cs.scores_, "P_Values": cs.pvalues_},index = x.columns)
feature_score.nlargest(n=61, columns="Score")


# Eliminate Categorical features with less or no relationship with target variable considering the p-value> 0.05

# In[42]:


#drop categoricalcolumns with less or no relationship with traget varibale
remove_columns = ['Ntm_Speciality','Gender','Risk_Low_Calcium_Intake','Risk_Segment_Prior_Ntm',
                  'Risk_Patient_Parent_Fractured_Their_Hip','Change_Risk_Segment','Risk_Untreated_Early_Menopause',
                  'Gluco_Record_Prior_Ntm','Risk_Family_History_Of_Osteoporosis','Risk_Osteogenesis_Imperfecta','Age_Bucket',
                  'Race','Risk_Segment_During_Rx','Ethnicity','Frag_Frac_Prior_Ntm']

d.drop(columns=remove_columns,inplace=True)               


# In[43]:


d.sample(10)


# In[44]:


d.shape


# In[45]:


# list of columns 
d.columns


# In[ ]:





# # Evaluation

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


X= d.drop(columns='Persistency_Flag')
Y = d['Persistency_Flag']


# In[48]:


Y.value_counts()


# ## Balance the dataset

# In[49]:


# balance the dataset using SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[50]:


sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X, Y)
print('Resampled dataset shape %s' % Counter(Y_res))


# In[51]:


Y_res.value_counts()


# In[52]:


# splitting dataset in 80% train dataset and 20% test dataset
X_train,X_test,Y_train,Y_test = train_test_split(X_res,Y_res, test_size=0.2,random_state=42)


# In[53]:


X_train.shape


# In[54]:


X_test.shape


# In[ ]:





# # Model Building

# ## Logistic Regression 

# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


reg=LogisticRegression()
reg.fit(X_train,Y_train) # Fit the model to the training data


# In[57]:


Y_pred=reg.predict(X_test) # Predict the classes on the test data
Y_pred


# In[58]:


np.mean(Y_pred==Y_test)


# In[59]:


pd.crosstab(Y_test,Y_pred)


# In[60]:


lreg_data=reg.score(X,Y)
lreg_train=reg.score(X_train,Y_train)
lreg_test=reg.score(X_test,Y_test)
print ("Accuracy of All dataset: " ,(lreg_data))
print ("Accuracy of Train dataset: " ,(lreg_train))
print ("Accuracy of Test dataset: " ,(lreg_test))


# In[ ]:





# ## RandomForestClassifier 

# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


clf = RandomForestClassifier(max_depth=3, random_state=42)
clf.fit(X_train,Y_train) # Fit the model to the training data


# In[63]:


Y1_pred=clf.predict(X_test) # Predict the classes on the test data
Y1_pred


# In[64]:


np.mean(Y1_pred==Y_test)


# In[65]:


pd.crosstab(Y_test,Y1_pred)


# In[66]:


rft_data=clf.score(X,Y)
rft_train=clf.score(X_train,Y_train)
rft_test=clf.score(X_test,Y_test)
print ("Accuracy of All dataset: " ,(rft_data))
print ("Accuracy of Train dataset: " ,(rft_train))
print ("Accuracy of Test dataset: " ,(rft_test))


# In[ ]:





# # KNeighborsClassifier

# In[67]:


from sklearn.neighbors import KNeighborsClassifier


# In[68]:


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train,Y_train) # Fit the model to the training data


# In[69]:


YK_pred=neigh.predict(X_test) # Predict the classes on the test data
YK_pred


# In[70]:


np.mean(YK_pred==Y_test)


# In[71]:


pd.crosstab(Y_test,YK_pred)


# In[72]:


knc_data=neigh.score(X,Y)
knc_train=neigh.score(X_train,Y_train)
knc_test=neigh.score(X_test,Y_test)
print ("Accuracy of All dataset: " ,(knc_data))
print ("Accuracy of Train dataset: " ,(knc_train))
print ("Accuracy of Test dataset: " ,(knc_test))


# Overfitting model!

# ## GradientBoostingClassifer

# In[73]:


from sklearn.ensemble import GradientBoostingClassifier


# In[74]:


model=GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=2, random_state=40)
model.fit(X_train,Y_train) # Fit the model to the training data


# In[75]:


Y2_pred=model.predict(X_test) # Predict the classes on the test data
Y2_pred


# In[76]:


np.mean(Y2_pred==Y_test)


# In[77]:


pd.crosstab(Y_test,Y2_pred)


# In[78]:


gbc_data=model.score(X,Y)
gbc_train=model.score(X_train,Y_train)
gbc_test=model.score(X_test,Y_test)
print ("Accuracy of All dataset: " ,(gbc_data))
print ("Accuracy of Train dataset: " ,(gbc_train))
print ("Accuracy of Test dataset: " ,(gbc_test))


# The score of train dataset is higher than test dataset which means its overfitting. let's do hyperparameter tuning for grid gradient boosting classifier model.

# In[ ]:





# # Metrics for Evaluation

# ##  Accuracy, Precision, Recall and F1-Score

# In[100]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:





# In[101]:


#LogisticRegression
print(classification_report(Y_test,Y_pred))


# In[102]:


confusion_matrix(Y_test,Y_pred)


# In[ ]:





# In[147]:


#RandomForestTreeClassifier
print(classification_report(Y_test,Y1_pred))


# In[103]:


confusion_matrix(Y_test,Y1_pred)


# In[ ]:





# In[166]:


#KNeighborsClassifier without hyperparameter tuning
print(classification_report(Y_test,YK_pred))


# In[104]:


confusion_matrix(Y_test,YK_pred)


# In[ ]:





# In[168]:


#GradientBoostingClassifier withouthyper parameter tuning
print(classification_report(Y_test,Y2_pred))


# In[106]:


confusion_matrix(Y_test,Y2_pred)


# In[ ]:





# ## Lift and Gain

# In[108]:


import scikitplot as skplt


# ### Logistic Regression

# In[109]:


# Predict the classes on the test data, and return the probabilities for each class
Y_proba = reg.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(Y_test, Y_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[110]:


skplt.metrics.plot_lift_curve(Y_test, Y_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[ ]:





# ### Random Forest Classifier

# In[111]:


# Predict the classes on the test data, and return the probabilities for each class
Y1_proba = clf.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(Y_test, Y1_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[112]:


skplt.metrics.plot_lift_curve(Y_test, Y1_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[ ]:





# ### KNN Classifier 

# In[122]:


# Predict the classes on the test data, and return the probabilities for each class
YK_proba = neigh.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(Y_test, YK_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[123]:


skplt.metrics.plot_lift_curve(Y_test, YK_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[ ]:





# ### gradient Boosting Classifier 

# In[126]:


# Predict the classes on the test data, and return the probabilities for each class
Y2_proba = model.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(Y_test, Y2_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[127]:


skplt.metrics.plot_lift_curve(Y_test, Y2_proba, figsize=(7, 5), title_fontsize=20, text_fontsize=18)
plt.show()


# In[ ]:





# ## KS Statistics and ROC-AUC Score

# In most binary classification problems we use the KS-2samp test and ROC AUC score as measurements of how well the model separates the predictions of the two different classes.
# The KS statistic for two samples is simply the highest distance between their two CDFs, so if we measure the distance between the positive and negative class distributions, we can have another metric to evaluate classifiers.
# The ROC AUC score goes from 0.5 to 1.0, while KS statistics range from 0.0 to 1.0

# In[130]:


from scipy import stats
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


# In[131]:


#Logistic Regression
# Fit the model to the training data
reg.fit(X_train,Y_train)
# Predict the classes on the test data
Y_pred=reg.predict(X_test)
# Predict the classes on the test data, and return the probabilities for each class
Y_proba = reg.predict_proba(X_test)


# In[ ]:





# In[132]:


#RandomForestClassifier
# Fit the model to the training data
clf.fit(X_train,Y_train)
# Predict the classes on the test data
Y1_pred=clf.predict(X_test)
# Predict the classes on the test data, and return the probabilities for each class
Y1_proba = clf.predict_proba(X_test)


# In[ ]:





# In[133]:


#KNeighborsClassifier without hyperparameter tuning 
# Fit the model to the training data
neigh.fit(X_train,Y_train)
# Predict the classes on the test data
YK_pred=neigh.predict(X_test)
# Predict the classes on the test data, and return the probabilities for each class
YK_proba = neigh.predict_proba(X_test)


# In[ ]:





# In[135]:


#BoostingGradientClassifier without hyperparameter tuning 
# Fit the model to the training data
model.fit(X_train,Y_train)
# Predict the classes on the test data
Y2_pred=model.predict(X_test)
# Predict the classes on the test data, and return the probabilities for each class
Y2_proba = model.predict_proba(X_test)


# In[ ]:





# In[137]:


def evaluate_ks_and_roc_auc(y_real, y_proba):
    # Unite both visions to be able to filter
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba[:, 1]
    
    # Recover each class
    class0 = df[df['real'] == 0]
    class1 = df[df['real'] == 1]
    
    ks = ks_2samp(class0['proba'], class1['proba'])
    roc_auc = roc_auc_score(df['real'] , df['proba'])
    
    print(f"KS: {ks.statistic:.4f} (p-value: {ks.pvalue:.3e})")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return ks.statistic, roc_auc


# In[ ]:





# In[138]:


print("Logistic Regression:")
ks_LR, auc_LR = evaluate_ks_and_roc_auc(Y_test, Y_proba)


# In[139]:


print("Random Forest classifier:")
ks_RFC, auc_RFC = evaluate_ks_and_roc_auc(Y_test, Y1_proba)


# In[140]:


print("KNeighbors classifier:")
ks_RFC, auc_RFC = evaluate_ks_and_roc_auc(Y_test, YK_proba)


# In[142]:


print("Gradient Boosting classifier:")
ks_GBC, auc_GBC = evaluate_ks_and_roc_auc(Y_test, Y2_proba)


# In[ ]:





# After considering score,classification report, confusion Matrix, lift anf gain curves, KS-Statistics and ROC-AUC score KNNC without Hyperparameter tuning is the best model

# # Save the Model

# In[144]:


# import pickle library
import pickle # its used for seriealizing and de-seriealizing a python object Structure
pickle.dump(neigh, open('model.pkl','wb'))       # open the file for writing
model = pickle.load(open('model.pkl','rb'))    # dump an object to file object


# In[146]:


print(model.predict([[0,2,1,0,3,5,0,1,0,0,0,0,0,1,1,1,2,1,2,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))


# In[ ]:





# In[ ]:




