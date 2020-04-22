#!/usr/bin/env python
# coding: utf-8

# In[229]:


#Importing libraries
import numpy as np
import pandas as pd


# In[230]:


#Uploading our treated dataset
data = pd.read_csv("D:/Rural Handmade/Question2/ecommerce_sample_dataset_cleaned.csv")


# In[231]:


data


# In[232]:


#Since most of the dataset is in textual form, we'll use NLP to get information.
import re
import nltk
from nltk import pos_tag, word_tokenize, PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
wordnet_lemmatizr=WordNetLemmatizer()
from termcolor import colored


# In[233]:


#Function returning Product types
def clean_product_type(dataframe):
    document=list(dataframe['product_category_tree'])
    product_types=[re.findall(r'\"(.*?)\"', sentence) for sentence in document]
    product_types=[' '.join(listed_items) for listed_items in product_types]
    return(product_types)


# In[234]:


#Function returning categories
def clean_categories(dataframe):
    document=list(dataframe['product_category_tree'].values)
    categories=[re.findall(r'name=(.*?)}',sentence) for sentence in document]
    categories=[' '.join(word) for word in categories]
    return(categories)


# In[235]:


#Function cleaning the document by removing special characters
def special_characters_cleaning(document):
    sentences=[]
    for sentence in document:
        sentences.append(re.sub('[^a-zA-Z0-9\n\.]',' ',str(sentence)))
    return(sentences)


# In[236]:


#Lemmatizing the document
def lemmetize_document(document):
    sentences=[]
    for sentence in document:
        word=[wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(sentence)]
        sentences.append(' '.join(words))
    return(sentences)


# In[237]:


#Extractio 
def categories_extraction(dataframe):
    categories=[word for item in dataframe['categories'] for word in item.split()]
    categories=list(set(categories))
    return(categories)


# In[238]:


def save_categories(dataframe):
    pass


# In[239]:


def pre_processing_document(document):
    document=special_characters_cleaning(document)
    document=lemmetize_document(document)
    document=[sentence.title() for sentence in document]
    return(document)


# In[240]:


def extract_categories_from_description(document,categories):
    extracted_categories=[]
    for sentence in document:
        extracted_categories.append(' '.join(set(categories).intersection(set(word_tokenize(sentence)))))
        return(extracted_categories)


# In[241]:


lemmetize= WordNetLemmatizer()
stemmer=PorterStemmer()


# In[242]:


#Applying our functions to dataframe
data["products"]=clean_product_type(data)


# In[243]:


data["categories"]=clean_categories(data)


# In[244]:


categories= list(set(data['product_category_tree'].values))
categories= [item.split() for item in data['product_category_tree']]
categories= [word.lower() for listed_item in categories for word in listed_item]
categories= list(set(categories))


# In[245]:


data["brand"]


# In[ ]:





# In[246]:


data


# In[247]:


#Creating a whole new column for the model pipeline.
data['detailed_description']= data['brand']+data['product_name']+ data["Category_of_Product"] + data["SecondaryCategory"] + data["TertiaryCategory"] + data["QuaternaryCategory"]


# In[248]:


data[data['Category_of_Product'] == "Clothing"]["detailed_description"]


# In[249]:


document= list(data['detailed_description'].values)
document= special_characters_cleaning(document)
document


# In[250]:


#Using TFIDF vector parameter
tfidf= TfidfVectorizer(stop_words= 'english', vocabulary= categories)
fit= tfidf.fit_transform(document)


# In[251]:


#Importing and initializing out NN Model
from sklearn.neighbors import NearestNeighbors
nn= NearestNeighbors(algorithm= 'brute', n_neighbors= 6).fit(fit)


# In[252]:


#Example no. 1 of recommendation.
#Recommendation based on Brand.

text= data[data['brand']== "FabHomeDecor"]['detailed_description'].values
result = nn.kneighbors(tfidf.transform(text))
for col in tfidf.transform(text).nonzero()[1]:
    print(tfidf.get_feature_names()[col], ' - ', tfidf.transform(text)[0, col])


# In[253]:


#Recommendations based on the search terms.
for item in result[1][0]:
    print(colored(data.iloc[item]['product_category_tree'].upper(), 'blue'), ':', document[item] + str(data["discounted_price"][item]))


# In[254]:


#Example no. 2

text= data[data['brand']== "Durian"]['detailed_description'].values
result = nn.kneighbors(tfidf.transform(text))
for col in tfidf.transform(text).nonzero()[1]:
    print(tfidf.get_feature_names()[col], ' - ', tfidf.transform(text)[0, col])


# In[255]:


#Recommendations based on the search terms.
for item in result[1][0]:
    print(colored(data.iloc[item]['product_category_tree'].upper(), 'blue'), ':', document[item] + str(data["discounted_price"][item]))


# In[256]:


#Example no.3

text= data[data['product_name']== "AW Bellies"]['detailed_description'].values
result = nn.kneighbors(tfidf.transform(text))
for col in tfidf.transform(text).nonzero()[1]:
    print(tfidf.get_feature_names()[col], ' - ', tfidf.transform(text)[0, col])


# In[257]:


#Recommendations based on the search terms.
for item in result[1][0]:
    print(colored(data.iloc[item]['product_category_tree'].upper(), 'blue'), ':', document[item] + str(data["discounted_price"][item]))


# In[260]:


#Example no.4

text= data[data["brand"] == "Alisha"]["detailed_description"].values
result = nn.kneighbors(tfidf.transform(text))
for col in tfidf.transform(text).nonzero()[1]:
    print(tfidf.get_feature_names()[col], ' - ', tfidf.transform(text)[0, col])


# In[261]:


#Recommendations based on the search terms.
for item in result[1][0]:
    print(colored(data.iloc[item]['product_category_tree'].upper(), 'blue'), ':', document[item] + str(data["discounted_price"][item]))


# In[264]:


#Example No. 5

text= data[data["brand"] == "Sicons"]["detailed_description"].values
result = nn.kneighbors(tfidf.transform(text))
for col in tfidf.transform(text).nonzero()[1]:
    print(tfidf.get_feature_names()[col], ' - ', tfidf.transform(text)[0, col])


# In[265]:


#Recommendations based on the search terms.
for item in result[1][0]:
    print(colored(data.iloc[item]['product_category_tree'].upper(), 'blue'), ':', document[item] + str(data["discounted_price"][item]))


# In[ ]:




