#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


train_df = pd.read_csv('./train_set.csv',sep='\t')


# In[3]:


test_df = pd.read_csv('./test_a.csv', sep ='\t')


# In[4]:


df = pd.concat((train_df,test_df))


# In[33]:


df['text'].values[0].split()[-2]


# In[52]:


anss = []
ans = ''
for text in df['text'].values:
    textline = text.strip().split()
    lenth = len(textline)
    #print(1)
    ans = ''
    for i in range(lenth): # line -> sentence
        if textline[i]=='900' or textline[i]=='2662' or textline[i]=='885':
            #print(textline[i],len(textline[i]))
            #break
            ans += textline[i]
            anss.append(ans)
            #print(ans)
            ans = '' 
        else:
            ans += (textline[i] + ' ')
            continue
    anss.append('sep')
        
   # print(textline[i])


# In[62]:


#import pandas 
#df = pd.DataFrame(anss,columns=['text'])
    


# In[84]:


#df.info()


# In[64]:


#df['len'] = df['text'].apply(lambda x: len(x.split()))


# In[83]:


#df[df['len']>256]


# In[80]:


'''
%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(x=df['len'],bins=2000)
plt.xlim([0,500])
plt.ylim([0,1000])
'''


# In[86]:


lenth = len(anss)
with open('./sentence.txt','w') as f:
    for index in range(lenth):
        if anss[index] == 'sep' or len(anss[index])==1:
            continue
        else:
            f.write(anss[index]+'\n')




