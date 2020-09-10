#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import os
import gc
from tensorflow.keras.layers import Layer
from gensim.models import Word2Vec
from tensorflow.keras.layers import (Bidirectional,
                                     Embedding,
                                     GRU, 
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D,
                                     Concatenate,
                                     SpatialDropout1D,
                                     BatchNormalization,
                                     Dropout,
                                     Dense,
                                     Activation,
                                     concatenate,
                                     Input
                                    )
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus),len(logical_gpus))
    except RuntimeError as e:
        print(e)

# In[ ]:


train_df = pd.read_csv('./split_train.csv')
valid_df = pd.read_csv('./split_test.csv')
test_df = pd.read_csv('./test_a.csv',sep='\t')
testb_df = pd.read_csv('./test_b.csv',sep='\t')


# In[ ]:


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=7000, lower=False,filters="")
tokenizer.fit_on_texts(list(train_df['text'].values)+list(valid_df['text'].values)+list(test_df['text'].values))

train_ = tokenizer.texts_to_sequences(train_df['text'].values)
valid_ = tokenizer.texts_to_sequences(valid_df['text'].values)
test_ = tokenizer.texts_to_sequences(testb_df['text'].values)


# In[ ]:


train_ = tf.keras.preprocessing.sequence.pad_sequences(train_, maxlen=2400)
valid_ = tf.keras.preprocessing.sequence.pad_sequences(valid_, maxlen=2400)
test_ = tf.keras.preprocessing.sequence.pad_sequences(test_, maxlen=2400)

word_vocab = tokenizer.word_index


# In[ ]:


# 200dim-CBoW
all_data=pd.concat([train_df['text'],valid_df['text'],test_df['text']])
file_name = './Word2Vec_200.model'
if not os.path.exists(file_name):
    model = Word2Vec([[word for word in document.split(' ')] for document in all_data.values],
                     size=200, 
                     window=5,
                     iter=10, 
                     workers=12, 
                     seed=2018, 
                     min_count=2)
    model.save(file_name)
else:
    model = Word2Vec.load(file_name)
print("add word2vec finished....") 


# In[ ]:


import gensim
Glove_model = gensim.models.KeyedVectors.load_word2vec_format('./Glove_200.txt',binary=False)


# In[ ]:


count = 0

embedding_matrix = np.zeros((len(word_vocab) + 1, 400))
for word, i in word_vocab.items():
    embedding_vector = np.concatenate((model.wv[word],Glove_model.wv[word])) if word in model else None
    if embedding_vector is not None:
        count += 1
        embedding_matrix[i] = embedding_vector
    else:
        unk_vec = np.random.random(400) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_matrix[i] = unk_vec


# In[ ]:


# 各个类别性能度量的函数
def category_performance_measure(labels_right, labels_pred):
    text_labels = list(set(labels_right))
    text_pred_labels = list(set(labels_pred))
    
    TP = dict.fromkeys(text_labels,0)  #预测正确的各个类的数目
    TP_FP = dict.fromkeys(text_labels,0)   #测试数据集中各个类的数目
    TP_FN = dict.fromkeys(text_labels,0) #预测结果中各个类的数目
    
    # 计算TP等数量
    for i in range(0,len(labels_right)):
        TP_FP[labels_right[i]] += 1
        TP_FN[labels_pred[i]] += 1
        if labels_right[i] == labels_pred[i]:
            TP[labels_right[i]] += 1
    #计算准确率P，召回率R，F1值
    for key in TP_FP:
        P = float(TP[key]) / float(TP_FP[key] + 1)
        R = float(TP[key]) / float(TP_FN[key] + 1)
        F1 = P * R * 2 / (P + R) if (P + R) != 0 else 0
        print("%s:\t P:%f\t R:%f\t F1:%f" % (key,P,R,F1))


# In[ ]:


Num_capsule=10
Dim_capsule=16
Routings=3


def squash(x, axis=-1):
    s_squared_norm = tf.keras.backend.sum(tf.keras.backend.square(x), axis, keepdims=True)
    scale = tf.keras.backend.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return x / scale


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = tf.keras.backend.conv1d(u_vecs, kernel=self.W)
        else:
            u_hat_vecs = tf.keras.backend.local_conv1d(u_vecs, kernel=self.W, kernel_size=[1], strides=[1])

        batch_size = tf.shape(u_vecs)[0]
        input_num_capsule = tf.shape(u_vecs)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs, [batch_size, input_num_capsule,self.num_capsule, self.dim_capsule])
        u_hat_vecs = tf.transpose(u_hat_vecs,perm=[0, 2, 1, 3])# final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        b = tf.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = tf.transpose(b, perm=[0, 2, 1])  # shape = [None, input_num_capsule, num_capsule] 
            c = tf.nn.softmax(b) # shape = [None, input_num_capsule, num_capsule] 
            c = tf.transpose(c, perm=[0, 2, 1])  # shape = [None, num_capsule, input_num_capsule] 
            s_j = tf.reduce_sum(tf.multiply(tf.expand_dims(c,axis=3) , u_hat_vecs) , axis=2)        
            outputs = self.activation(s_j) #[None,num_capsule,dim_capsule]
            if i < self.routings - 1:
                b = tf.reduce_sum(tf.multiply(tf.expand_dims(outputs,axis=2) , u_hat_vecs) , axis=3)
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[ ]:


def Gru_Capsule_Model(sent_length, embeddings_weight):
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=True)
    embed = SpatialDropout1D(0.2)(embedding(content))
    x = Bidirectional(GRU(400, return_sequences=True))(embed)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    capsule = Flatten()(capsule)
    x = Dense(1024)(capsule)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    output = Dense(14, activation="softmax")(x)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


# TextRNN+300CBoW-realcv
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
train_label = train_df['label'].values
from tensorflow.keras.utils import to_categorical
train_label = to_categorical(train_label)
#交叉验证的验证集的概率结果保存
#train_pre_matrix = np.zeros((train_df.shape[0],14)) #记录验证集的概率
#测试集的概率结果保存（cv次数，测试集的行数，标签）
test_pre_matrix = np.zeros((10,test_df.shape[0],14)) #将10轮的测试概率分别保存起来
#valid_pre_matrix = np.zeros((3,valid_df.shape[0],14)) #将10轮的测试概率分别保存起来
cv_scores=[] #每一轮线下的验证成绩
with tf.device("/gpu:1"):
    for i, (train_fold, test_fold) in enumerate(kf.split(train_,train_df['label'].values)):
        print("第%s的结果"%i)
        X_train, X_valid = train_[train_fold, :], train_[test_fold, :]
        y_train, y_valid = train_label[train_fold], train_label[test_fold]

        #在这里进行数据组装，
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
        val_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(128)
        valid_ds = tf.data.Dataset.from_tensor_slices((valid_,np.zeros((valid_.shape[0],14)))).batch(128)
        test_ds = tf.data.Dataset.from_tensor_slices((test_,np.zeros((test_.shape[0],14)))).batch(128)
        # 检查点保存至的目录
        checkpoint_dir = './TextCapsule_400_cv_advfinetune_checkpoints/cv_'+str(i)+'/'
        # 检查点的文件名
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        model = Gru_Capsule_Model(2400, embedding_matrix)

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=6)
        plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(checkpoint_prefix, monitor='val_accuracy', 
                                 verbose=2, save_best_only=True, mode='max',save_weights_only=True)

#        if not os.path.exists(checkpoint_dir):
#            model.fit(train_ds,
#                  epochs=5,
#                  validation_data=val_ds,
#                  callbacks=[early_stopping, plateau, checkpoint],
#                  verbose=1)

        model.load_weights(tf.train.latest_checkpoint( './TextCapsule_400_cv_advfinetune_checkpoints/cv_'+str(i)+'/'))
        #model.fit(train_ds,
        #          epochs=12,
        #          validation_data=val_ds,
        #          callbacks=[early_stopping, plateau, checkpoint],
        #          verbose=1)
    
       #验证集的结果 
        valid_prob = model.predict(val_ds)
        valid_pred = np.argmax(valid_prob,axis=1)
        y_valid = np.argmax(y_valid, axis=1)
        
        f1_score_ = f1_score(y_valid,valid_pred,average='macro') 
        print ("valid's f1-score: %s" %f1_score_)
        #train_pre_matrix[test_fold, :] =  valid_prob

        test_pre_matrix[i, :,:]= model.predict(test_ds)
        #valid_pre_matrix[i, :,:]= model.predict(valid_ds)
    #第一轮的ok
    #第二轮？
    #GPU 释放

        del model; gc.collect()#注意
        tf.keras.backend.clear_session()  #注意
        #if i==2:
        #    break
    
np.save("cv_TextCapsule_400advtestb_result.npy",test_pre_matrix)
#np.save("cv_TextCapsule_400lineoff_result.npy",valid_pre_matrix)


# In[ ]:


#validation = np.argmax(valid_pre_matrix.mean(axis=0),axis=1)
#f1 = f1_score(valid_df['label'].values,validation,average='macro') 
#print ("lineoff's f1-score: %s" %f1)
#category_performance_measure(list(valid_df['label']), list(validation))

