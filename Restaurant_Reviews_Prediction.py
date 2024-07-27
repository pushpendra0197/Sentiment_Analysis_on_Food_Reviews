#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np 
import pandas as pd 
import spacy
import nltk
import re 
import seaborn as sns
import string
from nltk.corpus import stopwords
stop=stopwords.words("english")
from nltk.stem import PorterStemmer
pos_stem=PorterStemmer()
from tkinter import *
import tkinter as tk
import tkinter.ttk as Combobox
from PIL import Image, ImageTk
import tkinter as tkinter
import joblib



# In[2]:


df=pd.read_csv(r"C:\Users\KINGNICKS-DELL\Desktop\Restuarant review\Restaurant_Reviews.csv")


# In[3]:


df


# In[4]:


df.drop_duplicates(inplace=True)


# In[5]:


df.duplicated().sum()


# In[6]:


df["Liked"]=df["Liked"].map({1:"Liked",0:"Disliked"})


# In[7]:


df


# In[8]:


df["Liked"].value_counts().plot(kind='bar')


# In[9]:


df["len"]=df["Review"].apply(len)


# In[10]:


df


# In[11]:


df.query('Liked=="Disliked"')["len"].mean()


# In[12]:


a=df.head(10).plot


# In[13]:


sns.barplot(x=df["Liked"],y=df["len"])


# In[14]:


def clean(text):
    review=text.lower()
    review=re.sub('[^a-zA-z]',' ',review)
    review=review.split()    
    review=[ i for i in review if i not in string.punctuation]
    review=[pos_stem.stem(word) for word in review]
    review=" ".join(review)
    
    return review
    


# In[15]:


df["cleaned_text"]=df["Review"].apply(lambda X:clean(X))


# In[16]:


df


# In[17]:


x=df["cleaned_text"]
y=df["Liked"]


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# In[20]:


cv=CountVectorizer(max_features=1500)


# In[21]:


cv.fit(x)


# In[22]:


X=cv.transform(x)


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=.80,random_state=0)


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


# In[26]:


LR=LogisticRegression()
DT=DecisionTreeClassifier()
RDF=RandomForestClassifier()
SVC=SVC()
GB=GaussianNB()


# In[27]:


LR.fit(xtrain,ytrain)


# In[28]:


print(accuracy_score(ytrain,LR.predict(xtrain)))
print(accuracy_score(ytest,LR.predict(xtest)))
print(classification_report(ytest,LR.predict(xtest)))


# In[29]:


DT.fit(xtrain,ytrain)


# In[30]:


print(accuracy_score(ytrain,DT.predict(xtrain)))
print(accuracy_score(ytest,DT.predict(xtest)))


# In[31]:


RDF.fit(xtrain,ytrain)


# In[32]:


print(accuracy_score(ytrain,RDF.predict(xtrain)))
print(accuracy_score(ytest,RDF.predict(xtest)))
print(classification_report(ytest,RDF.predict(xtest)))


# In[54]:


Review=Tk()
Review.title("Restaurant_Reviews_Prediction")
Review.geometry("1000x700")
Review.maxsize(height=700,width=1000)
Review.configure(bg="dodgerblue")
#labels and entrybox
label0=Label(Review,text="Restaurant_Reviews_Prediction",bg="DodgerBlue2",font=("TimesNewRomans",20),fg="black",bd=5,padx=15,pady=15)
label0.place(x=300,y=0)
label1=Label(Review,text="Enter Your Text-",bg="DodgerBlue2",font=("timesnewromans,12")).place(x=200,y=550)
e1=Entry(Review,bd=5,font=("timesnewromans",12),width=40)
e1.place(x=330,y=545)
e2=Entry(Review,width=5,font=("timesnewromans",20,"bold"),justify="center")
e2.place(x=430,y=170)
def check():
    p1=Entry.get(e1)
    print(p1)
    input=p1
    input_c=clean(input)
    input_vec=cv.transform([input_c])
    Model=joblib.load("Restaurant_Reviews_Prediction.pkl")
    prediction=Model.predict(input_vec)
    if prediction=="Disliked":
        Entry.insert(e2,0,"👎")
    if prediction=="Liked":
        Entry.insert(e2,0,"👍")    
def reset():
    e1.delete(0,END)
    e2.delete(0,END)
button=Button(Review,text="Submit",bg="red",font=("timesnewromans",9),bd=10,command=check)
button.place(x=200,y=620)
button=Button(Review,text="Reset",bg="red",font=("timesnewromans",9),bd=10,command=reset)
button.place(x=300,y=620)
Review.mainloop()


# In[ ]:




