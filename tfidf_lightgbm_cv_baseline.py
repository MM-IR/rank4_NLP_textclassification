import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('./data/train_set.csv',sep='\t')
test_df = pd.read_csv('./data/test_a.csv',sep='\t')

train_df['text_split'] = train_df['text'].apply(lambda x: str(x.split()))
test_df['text_split'] = test_df['text'].apply(lambda x: str(x.split()))

# text_split->tfidf
word_vec = TfidfVectorizer(analyzer='word',
            ngram_range=(1,2),#(1,3)
            min_df=3,  # 4  5
            max_df=0.9, # 0.95 1.0 
            use_idf=True,
            max_features = 3000,
            smooth_idf=True, 
            sublinear_tf=True)
train_term_doc = word_vec.fit_transform(train_df['text_split'])
test_term_doc = word_vec.transform(test_df['text_split'])

# F1 score-线下
from sklearn.metrics import f1_score
#[1,2,3,2,1,3]
#[1,2,3,1,1,3]
def cal_macro_f1(y_true,y_pred):
    score = f1_score(y_true,y_pred,average='macro')
    return score

X_train, X_eval, y_train, y_eval  = train_test_split(train_term_doc,train_df['label'],test_size=0.2,shuffle=True,random_state=2019) # split the training data

# CV
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=666)
train_matrix = np.zeros((train_df.shape[0],14)) #记录验证集的概率
##!!!!!
test_pre_matrix = np.zeros((10,test_df.shape[0],14)) #将5轮的测试概率分别保存起来
cv_scores=[] #每一轮线下的验证成绩

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
for i,(train_index,eval_index) in enumerate(kf.split(train_term_doc)):
    print(len(train_index),len(eval_index))
    
    
    #训练集
    X_train = train_term_doc[train_index]
    y_train = train_df['label'][train_index]
    
    #验证集
    X_eval = train_term_doc[eval_index]
    y_eval = train_df['label'][eval_index]
    
    # model = LogisticRegression(C=4, dual=False) 
    # model.fit(X_train,y_train)
    model =lgb.LGBMClassifier(boosting_type='gbdt', 
                   num_leaves=2**5,
                   max_depth=-1, 
                   learning_rate= 0.1,
                   n_estimators=2000,
                   objective='multiclass',
                   subsample=0.7,#
                   colsample_bytree=0.5,#
                   reg_lambda=10,#l2
                   n_jobs=16, #
                   num_class=19,#
                   silent=True, 
                   random_state=2019,
#                    class_weight=20,
                   colsample_bylevel=0.5,
                   min_child_weight=1.5,
                   metric='multi_logloss'
                  )
    model.fit(X_train,y_train,eval_set=(X_eval,y_eval), early_stopping_rounds=100)
    ####对于验证集进行预测
    eval_prob = model.predict_proba(X_eval)
    train_matrix[eval_index] = eval_prob.reshape((X_eval.shape[0], 14))#array
    
    eval_pred = np.argmax(eval_prob,axis=1)
    score = cal_macro_f1(y_eval,eval_pred)
    cv_scores.append(score)
    print("validation score is",score)
    
    ###对于测试集进行预测
    test_prob = model.predict_proba(test_term_doc)
    test_pre_matrix[i,:,:] = test_prob.reshape((test_term_doc.shape[0], 14))

all_pred = np.argmax(train_matrix,axis=1)
score = cal_macro_f1(train_df['label'],all_pred)
print("all validation score is",score)

test_pred = test_pre_matrix.mean(axis=0)
test_pred = np.argmax(test_pred,axis=1)
test_df['label'] = test_pred
test_df['label'].to_csv("submission_tfidf_lightGBM_10fold.csv",index=False,header=True,encoding='utf-8')

