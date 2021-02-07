#!/usr/bin/env python
# coding: utf-8

# Importing necessary packages
import re
import sys
import numpy as np
import pandas as pd

# xgboost algorithm
from xgboost import XGBClassifier

# packages for model development from scikit learn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# importing packages necessary for preprocessing from natural language toolkit(nltk)
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Declaring global variables
mbti_data = pd.DataFrame
fitted_vector = TfidfVectorizer()
models = {}

#  Preprocess posts 
#  - Removing URLs
#  - Remove MBTI personality type mentioned in text
#  - Remove special characters and numbers i.e keep only alphabet words with minimum of two    characters
#  - Lemmatize each word
def pre_process_posts(text):
    mbti_types = ['ENFJ' 'ENFP' 'ENTJ' 'ENTP' 'ESFJ' 'ESFP' 'ESTJ' 'ESTP' 'INFJ' 'INFP'
    'INTJ' 'INTP' 'ISFJ' 'ISFP' 'ISTJ' 'ISTP']

    # preloading stopwords
    stop_words = stopwords.words("english")

    # lemmatizer object
    lemmatizer = WordNetLemmatizer()
    # remove urls
    posts = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)
    #remove all other characters except alphabets
    posts = re.sub("[^a-zA-Z]", " ", posts)
    
    # remove MBTI types mentioned in posts
    for mbti_type in mbti_types:
        posts = posts.replace(mbti_type, "")
    
    posts = re.sub(" +", " ", posts).lower()
    
    # converting inflected words into base form i.e lemmatizing
    posts = " ".join([lemmatizer.lemmatize(word) for word in posts.split(' ') if word not in stop_words])
    return posts  

# tokenize text
def tokenize(text):
    tokens = [word for word in word_tokenize(text) if len(word) > 1]
    return tokens

def implement_base_model(X): 
    # encoding personality types

    enc = LabelEncoder()
    Y = enc.fit_transform(mbti_data["type"])

    # Splitting data to train and test sets.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Training the model
    print('Training base model.....')
    model = XGBClassifier(use_label_encoder=False)
    model.fit(X_train, Y_train, eval_metric='logloss')

    # Making predictions
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Accuracy of base model: ', round(accuracy * 100, 2), '%')

# Binarize/Encode MBTI to a vector
def convert_mbti_to_binary(personality):
    value_per_trait = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
    '''Calculate binary vector for MBTI personality string'''
    return [value_per_trait[t] for t in personality]

# Decode binary MBTI to personality string
def translate_binary_to_mbti(vector):
    '''Map binary vector to respective MBTI personality string'''
    dimension_traits = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
    p = ""
    for d, t in enumerate(vector):
        p += dimension_traits[d][t]
    return p

# Function to predict personality using previously built models from provided text.
def determine_mbti(text):
    global fitted_vector
    global models

    output_vector = []
    x = fitted_vector.transform(text)
    
    for model in models.values():
        prediction = model.predict(x)
        output_vector.append(prediction[0])
    
    return translate_binary_to_mbti(np.array(output_vector))

def main():
    global mbti_data
    global fitted_vector
    global models

    # importing data
    print('importing data...')
    mbti_data = pd.read_csv('dataset/mbti_kaggle.csv')

    tokenized_stop_words = word_tokenize(' '.join(stopwords.words('english')))

    print('preprocessing and vectorizing dataset...')
    # Converting plain texts post into TF/IDF vector representation to be fed to the model.
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words=tokenized_stop_words, preprocessor=pre_process_posts, tokenizer=tokenize, token_pattern=None)
    fitted_vector = tfidf_vectorizer.fit(mbti_data["posts"])
    X = fitted_vector.transform(mbti_data["posts"])

    # train a base model at first
    implement_base_model(X)

    # Now convert the personality of whole dataset to binary vectors.
    binarized_personalities = []

    for personality in mbti_data["type"]:
        binarized_personalities.append(convert_mbti_to_binary(personality))

    binarized_personalities = np.array(binarized_personalities)

    # Training four binary classifiers each predicting only a single trait for a dimension 
    # and save all of them in a dictionary.

    dimensions = [ 
                    ("energy", "IE: Introversion (I) / Extroversion (E)"), 
                    ("information", "NS: Intuition (N) / Sensing (S)"),
                    ("decision", "FT: Feeling (F) / Thinking (T)"),
                    ("organization", "JP: Judging (J) / Perceiving (P)")
                ]

    models = {}

    # setting up paramters for xgboost
    params = {}
    params['n_estimators'] = 200
    params['max_depth'] = 2
    params['nthread'] = 8
    params['learning_rate'] = 0.2

    # training models same way as before but with different paramters
    for i in range(len(dimensions)):
        print(f"Training model for { dimensions[i][0]} dimension..........")
        
        # personality type binary for current dimension only
        Y = binarized_personalities[:,i]
        
        # split data to train and test splits
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
        
        # fit model with training data
        model = XGBClassifier(use_label_encoder=False, **params)
        model.fit(X_train, Y_train, eval_metric='logloss')
        
        # add model to models dictionary
        models[dimensions[i][0]] = model
        
        # make prediction for test data
        Y_pred = model.predict(X_test)
        # evaluate prediction accuracy
        accuracy = accuracy_score(Y_test, Y_pred)
        print(f"* {dimensions[i][1]} Accuracy: {round(accuracy * 100.0, 2)}%")

    raw_text = ["""Getting started with data science and applying machine learning has never been as simple as it is now. There are many free and paid online tutorials and courses out there to help you to get started. I’ve recently started to learn, play, and work on Data Science & Machine Learning on Kaggle.com. In this brief post, I’d like to share my experience with the Kaggle Python Docker image, which simplifies the Data Scientist’s life.
    Awesome #AWS monitoring introduction.
    HPE Software (now @MicroFocusSW) won the platinum reader's choice #ITAWARDS 2017 in the new category #CloudMonitoring
    Certified as AWS Certified Solutions Architect 
    Hi, please have a look at my Udacity interview about online learning and machine learning,
    Very interesting to see the  lessons learnt during the HP Operations Orchestration to CloudSlang journey. http://bit.ly/1Xo41ci 
    I came across a post on devopsdigest.com and need your input: “70% DevOps organizations Unhappy with DevOps Monitoring Tools”
    In a similar investigation I found out that many DevOps organizations use several monitoring tools in parallel. Senu, Nagios, LogStach and SaaS offerings such as DataDog or SignalFX to name a few. However, one element is missing: Consolidation of alerts and status in a single pane of glass, which enables fast remediation of application and infrastructure uptime and performance issues.
    Sure, there are commercial tools on the market for exactly this use case but these tools are not necessarily optimized for DevOps.
    So, here my question to you: In your DevOps project, have you encountered that the lack of consolidation of alerts and status is a real issue? If yes, how did you approach the problem? Or is an ChatOps approach just right?
    You will probably hear more and more about ChatOps - at conferences, DevOps meet-ups or simply from your co-worker at the coffee station. ChatOps is a term and concept coined by GitHub. It's about the conversation-driven development, automation, and operations.
    Now the question is: why and how would I, as an ops-focused engineer, implement and use ChatOps in my organization? The next question then is: How to include my tools into the chat conversation?
    Let’s begin by having a look at a use case. The Closed Looped Incidents Process (CLIP) can be rejuvenated with ChatOps. The work from the incident detection runs through monitoring until the resolution of issues in your application or infrastructure can be accelerated with improved, cross-team communication and collaboration.
    In this blog post, I am going to describe and share my experience with deploying HP Operations Manager i 10.0 (OMi) on HP Helion Public Cloud. An Infrastructure as a Service platform such as HP Helion Public Cloud Compute is a great place to quickly spin-up a Linux server and install HP Operations Manager i for various use scenarios. An example of a good use case is monitoring workloads across public clouds such as AWS and Azure."""]
    personality = determine_mbti(raw_text)

    print('predicted personality for unseen text: ', personality)

if __name__ == '__main__':
    main()
