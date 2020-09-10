# coding: UTF-8
import time
import torch
import numpy as np
import os
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'News'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    for i in range(2,10):
        config.train_path = dataset + '/data/cv_'+str(i)+'/cv'+str(i)+'_train.txt'
        config.dev_path = dataset + '/data/cv_'+str(i)+'/cv'+str(i)+'_dev.txt' 
        config.test_path = dataset + '/data/cv_'+str(i)+'/cv_valid.txt'
        config.save_path = dataset + '/saved_dict/' + config.model_name + '512-'+str(i)+'.bin'      
        model = x.Model(config).to(config.device)
	
        train(config, model, train_iter, dev_iter, test_iter)
