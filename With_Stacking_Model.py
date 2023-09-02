import streamlit as st
import joblib 
import nltk
import string
from nltk.stem.porter  import PorterStemmer
import pickle
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB, MultinomialNB , BernoulliNB 
from sklearn.metrics import confusion_matrix , accuracy_score , precision_score , recall_score
import seaborn as sns
import matplotlib.pyplot as plt
ps=PorterStemmer() # This Function is used to convert Tenses into simple if Ing or E or etc is Connect with string so it will remove it and convert in Simple tense
MNB=MultinomialNB() # This Our model
from nltk.corpus import stopwords


def most_frequent(List):
    Count_Value=max(set(List), key = List.count)
    if Count_Value==0:
        print("This Mail is Hamm")
    elif Count_Value==1:
        print("THis Mail Is Spamm")
    else:
        print("What are Lappu Process")

    # We can Do this Process Directly with Model output but We should not done it because None of your Bussiness    




HEllo=input("\nEnter Text    ")

def transform_test(text):
    text=text.lower()
    text=nltk.word_tokenize(text) # In  This line Text converted in List After 
    y=[]
    for i in text:
        if i.isalnum(): # This Function Remove Special character like %$#@!
            y.append(i)
    text=y[:] # In This Line We Transform our List in Text variable
    y.clear()# in this line we CLear Our Y variable
    
    for i in text :
        if i not in string.punctuation and i not in stopwords.words("english"):
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
        
    
    return " ".join(y)
    
    
    
tfidf = joblib.load("/Users/jaychotaliya/Downloads/All_Coding_Program/Machine Learning With CampusX/17 Serving Models/17 in 3 Heroku/SMS-Learning/VectorizeinForm")
model = joblib.load("/Users/jaychotaliya/Downloads/All_Coding_Program/Machine Learning With CampusX/17 Serving Models/17 in 3 Heroku/SMS-Learning/StackingModel")
Voting = joblib.load("/Users/jaychotaliya/Downloads/All_Coding_Program/Machine Learning With CampusX/17 Serving Models/17 in 3 Heroku/SMS-Learning/VotingModel")
ETree = joblib.load("/Users/jaychotaliya/Downloads/All_Coding_Program/Machine Learning With CampusX/17 Serving Models/17 in 3 Heroku/SMS-Learning/ExtraTreesClassifier_Model")
"""
1.Preprocessing
2.Vectorize
3.Predict
4.Display
    """
    
#1.Preprocessing
Transformed_text=transform_test(HEllo)


#2.Vectorize

Vector_Input=tfidf.transform([Transformed_text]).toarray()


# #3.Predict

result=model.predict(Vector_Input)
print("Stacking Output",result)

Voting_result=Voting.predict(Vector_Input)
print("Voting Output",Voting_result)

CTreeee=ETree.predict(Vector_Input)
print("ExtraTreeClasifier Output",CTreeee)







#4.Display

List = []
if result[0]==0:
    List.append(0)
else:
    List.append(1)
    
if Voting_result[0]==0:
    List.append(0)
else:
    List.append(1)
    
if CTreeee[0]==0:
    List.append(0)
else:
    List.append(1)

print("All Model's Output",List)

most_frequent(List)


   

