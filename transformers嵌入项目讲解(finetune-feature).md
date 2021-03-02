# 是否求导就是利用好
with torch.no_grad():

# 1.利用Bert基于特征的方式进行建模
这里就是把输入数据经过Bert模型，来获取输入数据的特征，这些特征包含了整个句子的信息，是语境层面的。这种做法类似于ELMo的特征抽取。
**这里的BERT并没有参与后面的训练，仅仅进行特征抽取操作～**

model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


## 因为我们这里输入是英文句子，所以我们必须先将其分词，然后将其映射成词汇表中的索引，再喂给模型。（这个是按照wordpiece来为单位的）。
#add_special_tokens 表示在句子的首尾添加[CLS]和[END]符号
train_tokenized = train_set[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

这里就是获得了对应的encode tokenid～

## 接着就是将每个句子都处理成同一个长度，就是常见的PAD操作。我们在短的句子末尾添加一系列的PAD符号～

train_max_len = 0
for i in train_tokenized.values:
    if len(i) > train_max_len:
        train_max_len = len(i)

train_padded = np.array([i + [0] * (train_max_len-len(i)) for i in train_tokenized.values])
print("train set shape:",train_padded.shape)

## 然后我们还需要获得对应的attn——mask@针对每个sample中非pad部分
train_attention_mask = np.where(train_padded != 0, 1, 0)

## 接着就是直接获得特征的输出
train_input_ids = torch.tensor(train_padded).long()
train_attention_mask = torch.tensor(train_attention_mask).long()
with torch.no_grad():
    train_last_hidden_states = model(train_input_ids, attention_mask=train_attention_mask)

### 这里的输出就是
train_last_hidden_states[0].size()

output: torch.Size([3000, 66, 768])

可以看出来就是每一个位置都是一个embedding，我们一般用这些embedding中的第一个，因为是CLS的embedding，代表句子整体的信息（更具有代表性）

## 切分数据集@结合咱们的feature
train_features = train_last_hidden_states[0][:,0,:].numpy()
train_labels = train_set[1]
train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels)

然后就可以拿来训练啦。

# 2。模型中的那个就是基于微调获得词向量















————————————————
版权声明：本文为CSDN博主「程序员的自我反思」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/a553181867/article/details/105389757/
