# fasttext text retrieval
import fasttext
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize 

train_df = pd.read_csv('./split_train.csv')
valid_df = pd.read_csv('./split_test.csv')
test_df = pd.read_csv('./test_a.csv',sep='\t')

train_df['label_ft'] = '__label__' + train_df['label'].astype(str) #__label__number
valid_df['label_ft'] = '__label__' + valid_df['label'].astype(str) #__label__number

X_train = train_df['text']
y_train = train_df['label']
X_test = test_df['text']
X_valid = valid_df['text']
KF = StratifiedKFold(n_splits=5,random_state=666,shuffle=True)
#test_pred = np.zeros((5,X_test.shape[0],14),int) #这儿用的是1column
#for KF_index, (train_index, valid_index) in enumerate(KF.split(X_train,y_train)):
if True:
    #if KF_index==0 or KF_index==1:
    #    continue
    #print('第', KF_index+1, '折交叉验证开始...')
    '''
    train_df[['text','label_ft']].iloc[train_index].to_csv('fasttext_train_df.csv', header=None, index=False, sep='\t') #利用kfold进行数据的划分
    # 模型构建
    model = fasttext.train_supervised('fasttext_train_df.csv', lr=0.1, epoch=27, wordNgrams=5, 
                                      verbose=2, minCount=1, loss='hs')
    model.save_model('fasttext'+str(KF_index)+'.bin')
    # 模型预测
    '''
    clf = fasttext.load_model('fasttext'+str(0)+'.bin')
    ##val_pred = [int(model.predict(x)[0][0].split('__')[-1]) for x in X_train.iloc[valid_index]]
    ##print('Fasttext准确率为：',f1_score(list(y_train.iloc[valid_index]), val_pred, average='macro'))
    ##print(classification_report(list(y_train.iloc[valid_index]),val_pred,digits=4,target_names = ['科技', '股票', '体育', '娱乐', '时政', '社会', '教育',  '财经','家居',  '游戏',  '房产',  '时尚',  '彩票','星座']))
    # 保存测试集预测结果
    dev_feat = [clf.get_sentence_vector(x) for x in X_valid]
    dev_feat = np.vstack(dev_feat)
    dev_feat = normalize(dev_feat)
    train_feat = [clf.get_sentence_vector(x) for x in X_train]
    train_feat = np.vstack(train_feat)
    train_feat = normalize(train_feat)
    td_ids = np.dot(dev_feat,train_feat.T)
    td_pred = [train_df.iloc[np.argsort(x)[::-1][:10]]['label'].value_counts().index[0] for x in td_ids[:]]
    print('Fasttext准确率为：',f1_score(valid_df['label'].values, td_pred, average='macro'))
    print(classification_report(valid_df['label'].values,td_pred,digits=4,target_names = ['科技', '股票', '体育', '娱乐', '时政', '社会', '教育',  '财经','>    家居',  '游戏',  '房产',  '时尚',  '彩票','星座']))
    #test_pr = [model.predict_proba(x) for x in X_test]
    #print(test_pr[0])
    #break
    ##test_pred_ = [int(model.predict_proba(x)[0][0].split('__')[-1]) for x in X_test]
    ##test_pred = np.column_stack((test_pred, test_pred_))  # 将矩阵按列合并
    
    ##valid_pred = [int(model.predict(x)[0][0].split('__')[-1]) for x in X_test]
# 取测试集中预测数量最多的数
##preds = []
##for i, test_list in enumerate(test_pred):
##    preds.append(np.argmax(np.bincount(test_list)))
##preds = np.array(preds)   
##print(preds.shape)
##np.savez(
