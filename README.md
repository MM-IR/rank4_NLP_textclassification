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
   
  
