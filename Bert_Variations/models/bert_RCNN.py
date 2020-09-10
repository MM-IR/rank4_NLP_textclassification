# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert-RCNN'
        self.train_path = dataset + '/data/cv_2/cv2_train.txt'                                # 训练集
        self.dev_path = dataset + '/data/cv_2/cv2_dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/cv_2/cv_valid.txt'                                  # 测试集
        self.submit_path = dataset + '/data/submit_test_b.txt'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '2.bin'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 100000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 14                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 16                                          # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.bert_path = './bert-small'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 512
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 512                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 512
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
