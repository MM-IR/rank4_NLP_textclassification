## 2020阿里云tianchi零基础入门NLP比赛: rank4选手总结
https://tianchi.aliyun.com/competition/entrance/531810/introduction

-该比赛面向零基础小白，比赛赛题是匿名数据的文本分类比赛，所谓匿名数据也就是脱敏数据，文字是用数字来表示的，所以该比赛一个重点就是如今比较火的预训练模型: Bert系列可能没办法拿来直接使用，以及Word2Vec和GloVe等词向量也必须选手重新自己训练，所以如果是对整个流程不是很清楚的选手，很建议参加该比赛或者复盘比赛来进一步深入地学习。

```
环境配置:
pytorch
sklearn
gensim
Tensorflow2.0+
xgboost
lightgbm
tqdm
huggingface/transformers
```

## 比赛数据集简介
该比赛为14个类别的新闻文本分类比赛，本次比赛数据集分为线下的有标签训练数据以及A榜以及B榜的测试数据，选手们需要用线下的有标签训练数据来进行监督学习，最后用对应的模型在B榜上对应的成绩作为最终排名。

## 项目框架简介

```
adversial_validation.py # 对抗验证-检测训练数据与测试数据是否属于同一分布

-bert/ # tianchi开源Bert+BiLSTM+Attention版本, 简单加上伪标签，single fold线下结果有0.9597
  ├── README.md
  ├── bert-small # 我自己重新预训练了一个Bert-small, 我想如果尝试用medium和base, 应该还可以有一些提升
  │   ├── config.json # 这个是需要针对不同bert进行修改的config文件！！
  │   ├── pytorch_model.bin # Bert模型文件
  │   └── vocab.txt # 本人生成的, 做了一下min_count剔除了一些出现频次很少的word
  ├── bert_mini_lstm_pl.py
  ├── pl_ensemble_0.95.npy
  ├── split_test.csv # 本人用sklearn的StratifiedKFold进行的线下验证集划分-1:9
  ├── split_train.csv # 本人用sklearn的StratifiedKFold进行的线下训练集划分-1:9
  ├── test_a.csv # A榜测试数据
  ├── test_b.csv # B榜测试数据
  ├── tokenization.py 
  └── train_set.csv # 原始线下训练数据

-Bert_Variations/ # 本人私下Bert模型集合-验证了一些trick
  |-News/ # 这个就是本人做的txt版本训练数据+验证数据+测试数据, 通过简单的EDA, 用句号 问号 和 感叹号进行了分句
   |-pl_ensemble_0.95.npy #这个就是我直接用textbigru+bertbilstmatt做的A榜数据预测，选择了0.95作为threshold的伪标签数据---后悔没有早点用,效果是有提升的,但是对Bert提升不明显, 对于bigru/capsule的话蛮有作用的
  |-pytorch_pretrained/ # 就是一个pkg, pytorch_pretrained_bert好像是
  |-run*.py # 其实做的有些重复了, 个人没有用好ArgumentParser
  |-models/ #这里封装了我各种bert变体
  
-data/ # 这个就是对应的数据文件

-fasttext_train.py # 用文本匹配的方式做的fasttext模型

-GloVe_200.txt # 用Stanford开源的GloVe脚本跑出来的GloVe词向量

-Pretrain_Bert/ 
  |-bert_input.py # sentence.txt生成
  |-config.json # bert预训练config
  |-pretraining_args.py # 预训练核心代码1
  |-pytorch_pretrained_bert # 一个pkg
  |-run_pretraining.py # bert预训练核心代码2
  
-textbigru/
  |-TextbiGRU_400.py
  |-TextCapsule_400.py
  |-TextCNN_400.py
  
 -tfidf_lightgbm_cv_baseline.py # 0.945 baseline
 
 -tfidf.py # tfidf features
   
 -stacking.py # demo 
```

## how to run(一些模型文件没有上传，可能会报错，建议自己debug)
```
1. pretrain the bert: 
- cd Pretrain_Bert/
- python bert_input.py
- python run_pretraining.py

2. bertbilstm+attn:
- cd bert/
- python bert_mini_lstm_pl.py # 添加了伪标签, 如果要去掉, 把pl_ensemble_0.95.npy 有关的去掉就行

3. bert系列
- cd Bert_Variations/
- CUDA_VISIBLE_DEVICES=0 python run_*.py --model ** # (*-you know, **-表示对应的模型文件，比如bert_RNN.py就是bert_RNN)

4. textbigru/textcapsule/textcnn 
- cd textbigru/
- python x.py

5. fasttext_retrieval
- python fasttext_train.py

6. tfidf_baseline
- python tfidf_lightgbm_cv_baseline.py

7. stacking/blending-demo(具体的本人写在ipython里-没有保存emm-但本demo还是有较强的可复用性的)
- python stacking.py

```
## 多方案performance一览(本次比赛线上和线下的分数基本一致,但是本人测试可能类别分布上有一些差异)
|方案(无cv表示singlefold单模型)|线下验证结果f1 score|
|---|---|
|tfidf_lightgbm_cv|0.943~0.945|
|textbirgru+pl|0.959|
|textcnn-FC|0.943|
|bertbilstmattn|0.9597|
|bert系列没有特别多的记录|0.955+|
|bert_mini系列|0.951~0.952|
|bert_small系列没有特别多的记录|0.955+|
|fasttext-text retrieval|0.93|

## 融合测试
基本上textbigru_cv+bertbilstmattn (无pl) 此时也有0.969的成绩
加上pl其实就比较接近0.97了
后来我尝试了加上几个bert系列(后悔没有加上pl，否则可能还会提高) 结合tfidf做了一下对应lr, lightgbm, xgboost的stacking-B榜分数达到0.9702
总结: 其实我在线下验证集上达到了0.971, 但是我觉得可能B榜的类别分布与训练集不一样，所以我只有0.9702。

具体的细节可以关注我在Datawhale和知乎上的讲解。

