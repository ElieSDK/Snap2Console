#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv('/home/es/Documents/Project/VG/game_dataset_cleaned.csv')


# In[3]:


df.shape


# In[4]:


pd.set_option("display.max.columns", None)


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe(include=object)


# In[9]:


df['category'].value_counts()


# In[10]:


tg = df['genres'].value_counts().head(10)
tg


# In[11]:


themes = df['themes'].value_counts().head(20)
themes


# In[27]:


#Missing data per column (which features are incomplete).
df.isna().sum().sort_values(ascending=False)


# In[32]:


sns.heatmap(df.isna(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.show()


# In[12]:


plt.figure(figsize=(12,6))
plt.bar(themes.index, themes.values)
plt.xticks(rotation=45, ha='right')
plt.title('Most Popular Themes')
plt.show()


# In[13]:


#Let's remove Action
themes2 = themes.iloc[1:]

plt.bar(themes2.index, themes2.values)
plt.xticks(rotation=45, ha='right')
plt.title('Most Popular Themes')
plt.show()


# In[14]:


plt.bar(tg.index, tg.values)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Most Popular Genres')
plt.show()


# In[15]:


#Number of games released over time (per year)
gy = df['release_date'].str.slice(0,4).value_counts().head(25)
#gy_sorted = gy.sort_values(ascending=False)
gy_sorted = gy.sort_index(ascending=False)
gy_sorted = gy_sorted.iloc[1:] #remove 2025
gy_sorted


# In[16]:


plt.bar(gy_sorted.index, gy_sorted.values)
plt.xticks(rotation=45, ha='right')
plt.title('Number of games released over time (per year)')
plt.show()


# In[22]:


#Top developers and publishers by number of games. / main_developers	publishers
topp = df['publishers'].value_counts().head(20)
topp


# In[23]:


plt.bar(topp.index, topp.values)
plt.xticks(rotation=45, ha='right')
plt.title('Top developers and publishers by number of games')
plt.show()


# In[51]:


#Distribution of ratings and aggregated ratings.
ratings = df['rating'].value_counts()

# Bin ratings by 5
binned_counts = ratings.groupby(pd.cut(ratings.index, np.arange(0, 105, 5)), observed=False).sum()

# Plot
binned_counts.plot(kind='bar', figsize=(12,6))
plt.xlabel("Rating Range")
plt.ylabel("Count")
plt.title("Distribution of Ratings (Grouped by 5)")
plt.xticks(rotation=45)
plt.show()


# In[ ]:




