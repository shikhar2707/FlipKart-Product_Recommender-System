#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Importing Libraries
import pandas as pd
import numpy as np


# In[24]:


#Reading our dataset file
data = pd.read_csv("D:/Rural Handmade/Question2/ecommerce_sample_dataset.csv")


# In[25]:


data


# In[26]:


#List of colums in our data
data.columns


# In[27]:


#Dropping non existant values from our dataset
data.dropna(subset = ["brand"],inplace = True)
data.dropna(subset = ["retail_price"] , inplace = True)
data.dropna(subset = ["discounted_price"],inplace = True)
data


# In[28]:


#Deleting colums which are not useful for analysis and recommender system pipeline.
data = data.drop(['uniq_id','product_url', 'pid',
       'image', 'is_FK_Advantage_product', 'product_rating',
       'overall_rating'],axis = 1)
data 


# In[29]:


data.info()


# In[30]:


#Usimg product category tree to get the primary,secondary and tertiary category of the product
data['Category_of_Product'] = data['product_category_tree'].apply(lambda x: x.split('>>')[0][2:])


# In[31]:


data


# In[32]:


#Defining the lambda functions to get the categories of the product. Try function is used to prevent errors if the some categories do not exist.
def secondary(x):
    try:
        return x.split('>>')[1][1:]
    except IndexError:
        return 'None '
    
def tertiary(x):
    try:
        return x.split('>>')[2][1:]
    except IndexError:
        return 'None '
    
def quaternary(x):
    try:
        return x.split('>>')[3][1:]
    except IndexError:
        return 'None '


# In[33]:


data['SecondaryCategory'] = data['product_category_tree'].apply(secondary)
data['TertiaryCategory'] = data['product_category_tree'].apply(tertiary)
data['QuaternaryCategory'] = data['product_category_tree'].apply(quaternary)


# In[34]:


data


# In[21]:


#Dropping the column after its information was retrieved
data = data.drop('product_category_tree',axis = 1)


# In[22]:


data


# In[35]:


#Converting the timestamp to understandable data time format
data['crawl_timestamp'] = pd.to_datetime(data['crawl_timestamp'])
data['crawl_year'] = data['crawl_timestamp'].apply(lambda x: x.year)
data['crawl_month'] = data['crawl_timestamp'].apply(lambda x: x.month)


# In[36]:


data


# In[37]:


#Dropping after useful info is retrieved
data = data.drop("crawl_timestamp", axis = 1)


# In[38]:


data


# In[55]:


#Plotting some bar graphs to get some insights over the data.

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#1. Plot of Monthly sales count.
plt.figure(figsize=(4,3))
data.groupby('crawl_month')['crawl_month'].count().plot(kind='bar')
plt.title('Sales Count by Month',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Month',fontsize=12)
plt.ylabel('Sales Count',fontsize=12)
plt.show()
print(data.groupby('crawl_month')['crawl_month'].count())


#Monthly sales show that December is by far the most busy month for FlipKart Shopping spree.
#Oddly, Data is missing for July to November.


# In[57]:


#2. Most sales as per Product Category

plt.figure(figsize=(12,8))
data['Category_of_Product'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Category_of_Product',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Main Categories by Sales.\n')
print(data['Category_of_Product'].value_counts()[:10])

#Jewellery and Clothing are the most invloved sale on the flipkart market followed by Mobiles and Accessories, Automotive,Heome Decor and Furnishing.


# In[45]:


#Delving deep into the data
plt.figure(figsize=(12,8))
data['SecondaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Secondary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Secondary Categories by Sales.\n')
print(data['SecondaryCategory'].value_counts()[:10])

#Women's clothing and Necklaces and Chains are the most sold products.


# In[46]:


plt.figure(figsize=(12,8))
data['TertiaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Tertiary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Tertiary Categories by Sales.\n')
print(data['TertiaryCategory'].value_counts()[:10])


# In[50]:


#Most Expensive thing on the market
max_price = data['retail_price'].max()


# In[51]:


data[data['retail_price']==max_price]

#It is actually a dual seater Sofa worth 2,50,500rs.


# In[52]:


#Checking the average discount percentage Product wise
data['discount_%'] = round(((data['retail_price'] - data['discounted_price']) / data['retail_price'] * 100),1) 
data[['product_name','retail_price','discounted_price','discount_%']].head()


# In[53]:


ProductDiscount = pd.DataFrame(data.groupby('Category_of_Product').agg({
    'discount_%':[(np.mean)],
    'Category_of_Product':['count']
}))

ProductDiscount.columns = ['_'.join(col) for col in ProductDiscount.columns]
ProductDiscount = ProductDiscount.sort_values(by=['Category_of_Product_count'],ascending=False)[:20]


# In[54]:


#Plotting the discount by the category of Product.
plt.figure(figsize=(12,8))
ProductDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Product Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('Category_of_Product',fontsize=12)
plt.show()
print('Product Discount (Percentage)\n')
print(ProductDiscount['discount_%_mean'].sort_values(ascending=False)[:8])


# In[58]:


data


# In[59]:


#Saving the treated data as a new saved file for recommendation model
data.to_csv("D:/Rural Handmade/Question2/ecommerce_sample_dataset_cleaned.csv")


# In[ ]:




