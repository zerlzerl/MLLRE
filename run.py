from utils import *
from data_loader import *
import torch
import argparse
import numpy as np
import random
from model import SimilarityModel
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', default=0, type=int,
                        help='cuda device index, -1 means use cpu')
    parser.add_argument('--train_file', default='dataset/training_data.txt',
                        help='train file')
    parser.add_argument('--valid_file', default='dataset/val_data.txt',
                        help='valid file')
    parser.add_argument('--test_file', default='dataset/val_data.txt',
                        help='test file')
    parser.add_argument('--relation_file', default='dataset/relation_name.txt',
                        help='relation name file')
    parser.add_argument('--glove_file', default='dataset/glove.6B.300d.txt',
                        help='glove embedding file')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='word embeddings dimensional')
    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='BiLSTM hidden dimensional')
    parser.add_argument('--task_arrange', default='random',
                        help='task arrangement method')
    parser.add_argument('--rel_encode', default='glove',
                        help='relation encode method')
    parser.add_argument('--meta_method', default='reptile',
                        help='meta learning method, maml and reptile can be choose')
    parser.add_argument('--task_num', default=10, type=int,
                        help='number of tasks')
    parser.add_argument('--train_instance_num', default=100, type=int,
                        help='number of instances for one relation, -1 means all.')
    parser.add_argument('--step_size', default=0.1, type=float,
                        help='step size Epsilon')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--num_samplers', default=50, type=int,
                        help='number of samplers selected in one task')
    parser.add_argument('--random_seed', default=317, type=int,
                        help='random seed')



    opt = parser.parse_args()

    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    np.random.RandomState(opt.random_seed)

    device = torch.device(('cuda:%d' % opt.cuda_id) if torch.cuda.is_available() and opt.cuda_id >= 0 else 'cpu')

    # do following process
    split_train_data, split_test_data, split_valid_data, relation_numbers, rel_features, vocabulary, embedding = \
        load_data(opt.train_file, opt.valid_file, opt.test_file, opt.relation_file, opt.glove_file,
                            opt.embedding_dim, opt.task_arrange, opt.rel_encode, opt.task_num,
                            opt.train_instance_num)
    # prepare model
    inner_model = SimilarityModel(opt.embedding_dim, opt.hidden_dim, len(vocabulary),
                            np.array(embedding), 1, device)

    for task_index in range(opt.task_num):
        weights_before = deepcopy(inner_model.state_dict())

        train_task = split_train_data[task_index]
        test_task = split_test_data[task_index]
        valid_task = split_valid_data[task_index]


    if opt.meta_method == 'reptile':
        # use reptile to train model

    elif opt.meta_method == 'maml':
        # use reptile to train model, wait implement
        pass
    else:
        raise Exception('meta method %s not implement' % opt.meta_method)



if __name__ == '__main__':
    main()