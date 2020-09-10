'''

adversial validation:
this is to make sure that the training data and the test data come from the same distribution
-if the clf's result is great, the competition might not be suitable for you

'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
train_df = pd.read_csv('./bert/train_set.csv',sep='\t',nrows=5000)
test_df = pd.read_csv('./bert/test_a.csv',sep='\t',nrows=5000)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500).fit(train_df['text'].iloc[:].values)
train_tfidf = tfidf.transform(train_df['text'].iloc[:].values)
test_tfidf = tfidf.transform(test_df['text'].iloc[:].values)

train_test = np.vstack([train_tfidf.toarray(), test_tfidf.toarray()]) # new training data
lgb_data = lgb.Dataset(train_test, label=np.array([1]*5000+[0]*5000))
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['metric'] = 'auc'
result = lgb.cv(params, lgb_data, num_boost_round=100, nfold=3, verbose_eval=20)
print(pd.DataFrame(result))






