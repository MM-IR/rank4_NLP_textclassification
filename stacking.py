import os
import re
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
#from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,StratifiedKFold

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random

#导入数据
train=pd.read_csv("../data/train_set.csv")
test=pd.read_csv("../data/test_a.csv")

result=test[['id']].copy()
train_label=train['class'].values
lb = LabelEncoder()
lb.fit(train['class'].values)

path="./"
#
capsule_lstm_train_char=np.load(path+"capsule_lstm10_article.npz")['train']
#
capsule_lstm_test_char=np.load(path+"capsule_lstm10_article.npz")['test']

#词
capsule_lstm_train_word=np.load(path+"capsule_lstm10_word_seg.npz")['train']
#词
capsule_lstm_test_word=np.load(path+"capsule_lstm10_word_seg.npz")['test']

gru3_train_word=np.load(path+"get_text_gru310_word_seg.npz")['train']
gru3_test_word=np.load(path+"get_text_gru310_word_seg.npz")['test']

gru4_train_word=np.load(path+"get_text_gru410_word_seg.npz")['train']
gru4_test_word=np.load(path+"get_text_gru410_word_seg.npz")['test']

x_train=np.concatenate([capsule_lstm_train_char,
                           capsule_lstm_train_word,
                           gru3_train_word,
                           gru4_train_word
                       ],axis=1)

x_test=np.concatenate([capsule_lstm_test_char,
                           capsule_lstm_test_word,
                           gru3_test_word,
                           gru4_test_word
                         ],axis=1)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=520).split(train['id'])
fold_index=train[['id']].copy()
for i, (train_fold, test_fold) in enumerate(kf):
    fold_index.loc[test_fold,'fold']=int(i)
fold_index['fold']=fold_index['fold'].astype(int)
fold_index.to_csv('fold_index.csv',index=False) #标清楚某个数据-id对应的是第几个fold

nb_classes =19
dims = x_train.shape[1]
epochs = 15
# parameter grids
param_grid = [
     #(1, 6, 0.73, 0.756, 0.00001, 0.017, 2400),
     # (1, 8, 0.789, 0.97, 0, 0.018, 1100),
     #(1, 5, 0.7, 0.7, 0.001, 0.01, 1500),
     # (1, 6, 0.89, 0.994, 0.0001, 0.02421, 700),
     #(1, 10, 0.74, 0.908, 0.0005, 0.0141, 1750),
     #(1, 15, 0.7890, 0.890643, 0.231, 0.21, 900),
     #(1, 19, 0.78, 0.97453, 0.00009, 0.01, 3900),
     #(1, 6, 0.71, 0.71, 0, 0.01, 1250),
      #(1, 8, 0.77, 0.83, 0.001, 0.03, 900),
     (1, 3, 0.7, 0.7, 0.00008, 0.01, 300),
     #(1, 8, 0.824, 0.0241, 0.000177,0.02406 ,743)        

# kb8:  (1, 10, 0.87, 0.88, 0.000429, 0.029963, 652)        
# kb9:  (1, 8, 0.824, 0.0241, 0.000177,0.02406 ,743 )         

]



from sklearn.metrics import f1_score
import xgboost as xgb
xfolds = pd.read_csv('fold_index.csv')
# work with 5-fold split

fold_index = xfolds.fold
n_folds = len(np.unique(fold_index))
train_model_pred = np.zeros((x_train.shape[0], 19))
test_model_pred = np.zeros((x_test.shape[0], 19))
for i in range(len(param_grid)):
    print("processing parameter combo:", param_grid[i])
    # configure model with j-th combo of parameters
    x = param_grid[i]
    clf = xgb.XGBClassifier(objective='multi:softmax',
                            n_estimators=x[6],
                            max_depth=x[1],
                            min_child_weight=x[0],
                            learning_rate=x[5],
                            silent=True,
                            subsample=x[3],
                            colsample_bytree=x[2],
                            gamma=x[2],
                            seed=6666,
                            num_class=19,
                            n_jobs=10)
    for j in range(0,n_folds):
        idx0 = np.where(fold_index != j)
        idx1 = np.where(fold_index == j)  
        x0 = np.array(x_train)[idx0,:][0]
        x1 = np.array(x_train)[idx1,:][0]
        y0 = np.array(train_label)[idx0]
        y1 = np.array(train_label)[idx1]
        clf.fit(x0, y0, eval_metric="mlogloss", eval_set=[(x0, y0),(x1, y1)],early_stopping_rounds=5,verbose=100)

        train_model_pred[idx1, :] =  clf.predict_proba(x1)
        test_model_pred +=clf.predict_proba(x_test)
        print ("valid's macro-f1: %s" % f1_score(y1.reshape(-1,1), 
                                                lb.inverse_transform(np.argmax( clf.predict_proba(x1), 1)).reshape(-1,1),
                                                              average='macro'))

        print("finished fold:", j)


print ("offline test score: %s" % f1_score(train_label.reshape(-1,1), 
                                      lb.inverse_transform(np.argmax(train_model_pred, 1)).reshape(-1,1),
                                      average='micro'))

clf.fit(x_train, train_label, eval_metric="mlogloss",verbose=100)
test_model_pred =clf.predict_proba(x_test)




np.savez('stacking_offline8130664763338775.npz', train=train_model_pred, test=test_model_pred)
