# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert-cnn'
        self.train_path = dataset + '/data/cv_0/cv0_train.txt'                                # 训练集
        self.dev_path = dataset + '/data/cv_0/cv0_dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/cv_0/cv_valid.txt'                                  # 测试集
        self.submit_path = dataset + '/data/submit_test.txt'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '512-0.bin'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 100000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 14                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.bert_path = './bert-small/'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 512
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 512                                          # 卷积核数量(channels数)
        self.dropout = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out
