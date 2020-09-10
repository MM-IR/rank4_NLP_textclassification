import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('./train_set.csv', sep='\t')
test_df = pd.read_csv('./test_a.csv',sep='\t')

train_df['len_text'] = train_df['text'].apply(lambda x: len(x.split()))
test_df['len_text'] = train_df['text'].apply(lambda x: len(x.split()))
# length
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax = plt.hist(x=train_df['len_text'],bins=100,label='train')
ax = plt.hist(x=test_df['len_text'],bins=100,label='test')
plt.axis('tight')
plt.xlabel("length of sample")
plt.ylabel("count")
plt.legend()
# seaborn length
import seaborn as sns
plt.figure(figsize=(15,5))
ax = sns.distplot(train_df['len_text'],bins=100)
ax = sns.distplot(test_df['len_text'],bins=100)
plt.axis('tight')
plt.xlabel('length of sample')
plt.ylabel('count')
plt.legend()

