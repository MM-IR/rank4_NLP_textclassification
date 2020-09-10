# coding: UTF-8
import time
import torch
import numpy as np
import os
from train_eval import train, init_network, test
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
    new_data = np.load('./pl_ensemble_0.95.npy')
    #new_data_x = test_df.iloc[new_data[:,0]].text.values
    new_data_y = new_data[:,1]

#    start_time = time.time()
#    print("Loading data...")
#    train_data, dev_data, test_data = build_dataset(config)
#    train_iter = build_iterator(train_data, config)
#    dev_iter = build_iterator(dev_data, config)
#    test_iter = build_iterator(test_data, config)
#    time_dif = get_time_dif(start_time)
#    print("Time usage:", time_dif)

    # train
    for i in range(0,5):
        config.train_path = dataset + '/data/fold5/cvfold'+str(i)+'_train.txt'
        config.dev_path = dataset + '/data/fold5/cvfold'+str(i)+'_dev.txt' 
        config.test_path = dataset + '/data/submit_test.txt'
        config.save_path = dataset + '/saved_dict/' + config.model_name + 'adv512-5fold-'+str(i)+'.bin'  
        train_data, dev_data, test_data, submit_data = build_dataset(config)
        temp_data = []
        for idx in new_data[:,0]:
            temp_data.append(list(test_data[idx]))
        for i in range(len(temp_data)):
            temp_data[i][1] = new_data_y[i]
            temp_data[i] = tuple(temp_data[i])
        train_data.extend(temp_data)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        submit_iter = build_iterator(submit_data, config)    
        model = x.Model(config).to(config.device) 
        #if i == 0:
        #    pass
        #else:
        train(config, model, train_iter, dev_iter, test_iter)
        #test(config, model, test_iter,'bertdrop_valid_'+str(i)+'.npy')
        #test(config, model, submit_iter, 'bertdrop_submit_'+str(i)+'.npy')
        #test(config, model, dev_iter, 'bertdrop_train_'+str(i)+'.npy')
