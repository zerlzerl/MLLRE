import math
import pickle
import sys
import time

from tqdm import tqdm

from utils import *
from data_loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
from model import SimilarityModel
from copy import deepcopy
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def feed_samples(model, samples, loss_function, all_relations, device,
                 alignment_model=None):
    """

    :param model: SimilarityModel
    :param samples: 一个batch的训练数据
    :param loss_function: MarginLoss：计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0
    :param all_relations: 全部关系包括/fill/fill/fill的word list的list。 = [[rel_0_word_indices], [rel_1_word_indices], ..., [rel_80_word_indices]]
    :param device:
    :param alignment_model:
    :return:
    """
    questions, relations, relation_set_lengths = process_samples(
        samples, all_relations, device)  # 将每个sample都进行展开，做成question和一个候选关系一对一的形式，relation_set_lengths记录了每个sample展开成了几个句子
    ranked_questions, alignment_question_indexs = \
        ranking_sequence(questions)  # 输入一个一维tensor的list，对其中的每一个list，按其中元素的长度进行排序，从大到小排序，返回排序后的list和对应原序列中的index
    ranked_relations, alignment_relation_indexs = \
        ranking_sequence(relations)
    question_lengths = [len(question) for question in ranked_questions]  # 排序之后每个question list中句子的长度
    relation_lengths = [len(relation) for relation in ranked_relations]  # 排序之后每个relation list中句子的长度
    pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)  # 进行补齐
    pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
    pad_questions = pad_questions.to(device)
    pad_relations = pad_relations.to(device)

    model.zero_grad()
    if alignment_model is not None:
        alignment_model.zero_grad()
    model.init_hidden(device, sum(relation_set_lengths))
    all_scores = model(pad_questions, pad_relations, device,
                       alignment_question_indexs, alignment_relation_indexs,
                       question_lengths, relation_lengths, alignment_model)  # 每个句子和关系对的similarity score
    all_scores = all_scores.to('cpu')
    pos_scores = []
    neg_scores = []
    pos_index = []
    start_index = 0
    for length in relation_set_lengths:
        pos_index.append(start_index)
        pos_scores.append(all_scores[start_index].expand(length-1))
        neg_scores.append(all_scores[start_index+1:start_index+length])
        start_index += length
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)
    alignment_model_criterion = nn.MSELoss()

    loss = loss_function(pos_scores, neg_scores,
                         torch.ones(sum(relation_set_lengths)-
                                    len(relation_set_lengths)))
    loss.backward()
    return all_scores, loss

def evaluate_model(model, testing_data, batch_size, all_relations, device,
                   reverse_model=None):
    """

    :param model:
    :param testing_data:
    :param batch_size:
    :param all_relations:
    :param device:
    :param reverse_model:
    :return:
    """
    #print('start evaluate')
    num_correct = 0
    #testing_data = testing_data[0:100]
    for i in range((len(testing_data)-1)//batch_size+1):
        samples = testing_data[i*batch_size:(i+1)*batch_size]
        gold_relation_indexs, questions, relations, relation_set_lengths = \
            process_testing_samples(samples, all_relations, device)
        model.init_hidden(device, sum(relation_set_lengths))
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        ranked_relations, reverse_relation_indexs = \
            ranking_sequence(relations)
        question_lengths = [len(question) for question in ranked_questions]
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths, reverse_model)
        start_index = 0
        pred_indexs = []
        #print('len of relation_set:', len(relation_set_lengths))
        for j in range(len(relation_set_lengths)):
            length = relation_set_lengths[j]
            cand_indexs = samples[j][1]
            pred_index = (cand_indexs[
                all_scores[start_index:start_index+length].argmax()])
            if pred_index == gold_relation_indexs[j]:
                num_correct += 1
            #print('scores:', all_scores[start_index:start_index+length])
            #print('cand indexs:', cand_indexs)
            #print('pred, true:',pred_index, gold_relation_indexs[j])
            start_index += length
    #print(cand_scores[-1])
    #print('num correct:', num_correct)
    #print('correct rate:', float(num_correct)/len(testing_data))
    return float(num_correct)/len(testing_data)

def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' %num)
    print('')

def update_rel_cands(memory_data, all_seen_cands, num_cands):
    if len(memory_data) >0:
        for this_memory in memory_data:
            for sample in this_memory:
                valid_rels = [rel for rel in all_seen_cands if rel!=sample[0]]
                sample[1] = random.sample(valid_rels, min(num_cands, len(valid_rels)))

def offset_list(l, offset):
    if offset == 0:
        return l
    offset_l = [None] * len(l)
    for i in range(len(l)):
        offset_l[(i + offset) % len(l)] = l[i]

    return offset_l

def resort_list(l, index):
    resorted_l = [None] * len(l)
    for i in range(len(index)):
        resorted_l[i] = l[index[i]]

    return resorted_l

def resort_memory(memory_pool, similarity_index):
    memory_pool = sorted(memory_pool, key=lambda item: np.argwhere(similarity_index == item[0]))
    return memory_pool
    # 隔k个取一个
    # _memo_pool = []
    # for i in range(0, len(memory_pool), 2):
    #     _memo_pool.append(memory_pool[i])
    # return _memo_pool


# get relation embedding of current seen relations
def tsne_relations(model, seen_task_relations, all_relations, device, task_idx, alignment_model=None, before_alignment=False):
    color_schema = ['black', 'darkviolet', 'firebrick', 'green', 'gold',
                      'chartreuse', 'darkorange', 'chocolate', 'cyan', 'grey']
    task_labels = ['Task %d' % idx for idx in task_idx]
    # get relation embeddings of current seen relations
    current_seen_relations = []
    relation_cluster = []
    for i in range(len(seen_task_relations)):
        current_seen_relations.extend(seen_task_relations[i])
        relation_cluster.extend([i] * len(seen_task_relations[i]))

    relations_index = []
    for rel in current_seen_relations:
        relations_index.append(torch.tensor(all_relations[rel - 1], dtype=torch.long).to(device))

    model.init_hidden(device, len(relations_index))
    ranked_relations, alignment_relation_indexs = ranking_sequence(relations_index)
    relation_lengths = [len(relation) for relation in ranked_relations]

    pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)

    rel_embeds = model.compute_rel_embed(pad_relations, relation_lengths,
                                         alignment_relation_indexs,
                                         alignment_model, before_alignment)
    rel_embeds = rel_embeds.detach().cpu().numpy()

    # # draw tsne picture
    # X_tsne = TSNE(n_components=2, random_state=33).fit_transform(rel_embeds)
    # task_label_cords_list = [None] * len(seen_task_relations)
    # for i in range(len(current_seen_relations)):
    #     relation_idx = current_seen_relations[i]
    #     rel_cluster = relation_cluster[i]
    #     relation_cord = X_tsne[i]
    #     relation_color = color_schema[rel_cluster]
    #     plt.scatter(relation_cord[0], relation_cord[1], alpha=0.6, marker='o', c=relation_color)
    #     # plt.text(relation_cord[0], relation_cord[1] + 1.0, str(relation_idx), c=relation_color)
    #
    #     if task_label_cords_list[rel_cluster] is None:
    #         task_label_cords_list[rel_cluster] = [relation_cord]
    #     else:
    #         task_label_cords_list[rel_cluster].append(relation_cord)
    #
    # # add task label
    # for i in range(len(task_label_cords_list)):
    #     task_label_cords = task_label_cords_list[i]
    #     task_label_cord = np.mean(np.array(task_label_cords), axis=0)
    #     plt.text(task_label_cord[0], task_label_cord[1] + 2.0, task_labels[i], c=color_schema[i])
    #
    # plt.title('Relation embedding distance t-SNE plot after %d tasks trained' % len(seen_task_relations),
    #           fontsize='large', fontweight='bold', color='black')
    # plt.show()

    return rel_embeds



def main(opt):

    print(opt)
    # print('线性outer step formula，0.6 step size， 每task聚类取50个memo')
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    np.random.RandomState(opt.random_seed)
    start_time = time.time()
    checkpoint_dir = os.path.join(opt.checkpoint_dir, '%.f' % start_time)

    device = torch.device(('cuda:%d' % opt.cuda_id) if torch.cuda.is_available() and opt.cuda_id >= 0 else 'cpu')

    # do following process
    split_train_data, train_data_dict, split_test_data, split_valid_data, relation_numbers, rel_features, \
    split_train_relations, vocabulary, embedding = \
        load_data(opt.train_file, opt.valid_file, opt.test_file, opt.relation_file, opt.glove_file,
                  opt.embedding_dim, opt.task_arrange, opt.rel_encode, opt.task_num,
                  opt.train_instance_num)
    print('\n'.join(['Task %d\t%s' % (index, ', '.join(['%d' % rel for rel in split_train_relations[index]])) for index in range(len(split_train_relations))]))

    # offset tasks
    # split_train_data = offset_list(split_train_data, opt.task_offset)
    # split_test_data = offset_list(split_test_data, opt.task_offset)
    # split_valid_data = offset_list(split_valid_data, opt.task_offset)
    # task_sq = [None] * len(split_train_relations)
    # for i in range(len(split_train_relations)):
    #     task_sq[(i + opt.task_offset) % len(split_train_relations)] = i
    # print('[%s]' % ', '.join(['Task %d' % i for i in task_sq]))

    # insert 6th-task
    # task_index = [[6, 0, 1, 2, 3, 4, 5, 7, 8, 9],
    #               [0, 6, 1, 2, 3, 4, 5, 7, 8, 9],
    #               [0, 1, 6, 2, 3, 4, 5, 7, 8, 9],
    #               [0, 1, 2, 6, 3, 4, 5, 7, 8, 9],
    #               [0, 1, 2, 3, 6, 4, 5, 7, 8, 9],
    #               [0, 1, 2, 3, 4, 6, 5, 7, 8, 9],
    #               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #               [0, 1, 2, 3, 4, 5, 7, 6, 8, 9],
    #               [0, 1, 2, 3, 4, 5, 7, 8, 6, 9],
    #               [0, 1, 2, 3, 4, 5, 7, 8, 9, 6]]

    task_sequence = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                  [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                  [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                  [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                  [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                  [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                  [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                  [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]


    split_train_data = resort_list(split_train_data, task_sequence[opt.sequence_index])
    split_test_data = resort_list(split_test_data, task_sequence[opt.sequence_index])
    split_valid_data = resort_list(split_valid_data, task_sequence[opt.sequence_index])
    split_train_relations = resort_list(split_train_relations, task_sequence[opt.sequence_index])
    print('[%s]' % ', '.join(['Task %d' % idx for idx in task_sequence[opt.sequence_index]]))
    kl_dist_ht = read_json(opt.kl_dist_file)

    # tmp = [[0, 1, 2, 3], [1, 0, 4, 6], [2, 4, 0, 5], [3, 6, 5, 0]]
    sorted_similarity_index = np.argsort(np.asarray(kl_dist_ht), axis=1) + 1


    # prepare model
    inner_model = SimilarityModel(opt.embedding_dim, opt.hidden_dim, len(vocabulary),
                                  np.array(embedding), 1, device)

    memory_data = []
    memory_pool = []
    memory_question_embed = []
    memory_relation_embed = []
    sequence_results = []
    result_whole_test = []
    seen_relations = []
    all_seen_relations = []
    rel2instance_memory = {}
    memory_index = 0
    seen_task_relations = []
    rel_embeddings = []
    for task_ix in range(opt.task_num):  # outside loop
        # reptile start model parameters pi
        weights_before = deepcopy(inner_model.state_dict())

        train_task = split_train_data[task_ix]
        test_task = split_test_data[task_ix]
        valid_task = split_valid_data[task_ix]
        train_relations = split_train_relations[task_ix]
        seen_task_relations.append(train_relations)

        # collect seen relations
        for data_item in train_task:
            if data_item[0] not in seen_relations:
                seen_relations.append(data_item[0])

        # remove unseen relations
        current_train_data = remove_unseen_relation(train_task, seen_relations)
        current_valid_data = remove_unseen_relation(valid_task, seen_relations)
        current_test_data = []
        for previous_task_id in range(task_ix + 1):
            current_test_data.append(remove_unseen_relation(split_test_data[previous_task_id], seen_relations))

        for this_sample in current_train_data:
            if this_sample[0] not in all_seen_relations:
                all_seen_relations.append(this_sample[0])

        update_rel_cands(memory_data, all_seen_relations, opt.num_cands)

        # train inner_model
        loss_function = nn.MarginRankingLoss(opt.loss_margin)
        inner_model = inner_model.to(device)
        optimizer = optim.Adam(inner_model.parameters(), lr=opt.learning_rate)
        t = tqdm(range(opt.outside_epoch))
        best_valid_acc = 0.0
        early_stop = 0
        best_checkpoint = ''

        #
        resorted_memory_pool = []
        for epoch in t:
            batch_num = (len(current_train_data) - 1) // opt.batch_size + 1
            total_loss = 0.0
            target_rel = -1
            for batch in range(batch_num):
                batch_train_data = current_train_data[batch * opt.batch_size: (batch + 1) * opt.batch_size]

                if len(memory_data) > 0:
                    # curriculum select and organize memory
                    # if target_rel == -1 or len(resorted_memory_pool) == 0:
                    #     target_rel = batch_train_data[0][0]
                    #     target_rel_sorted_index = sorted_similarity_index[target_rel - 1]
                    #     resorted_memory_pool = resort_memory(memory_pool, target_rel_sorted_index)
                    #
                    # if len(resorted_memory_pool) >= opt.task_memory_size:
                    #     current_memory = resorted_memory_pool[:opt.task_memory_size]
                    #     resorted_memory_pool = resorted_memory_pool[opt.task_memory_size + 1:]  # 更新剩余的memory
                    #     batch_train_data.extend(current_memory)
                    # else:
                    #     current_memory = resorted_memory_pool[:]
                    #     resorted_memory_pool = []  # 更新剩余的memory
                    #     batch_train_data.extend(current_memory)

                    # 淘汰的做法
                    # if len(resorted_memory_pool) != 0:
                    #     current_memory = resorted_memory_pool[:opt.task_memory_size]
                    #     resorted_memory_pool = resorted_memory_pool[opt.task_memory_size + 1:]  # 更新剩余的memory
                    #     batch_train_data.extend(current_memory)
                    # else:
                    #     target_rel = batch_train_data[0][0]
                    #     target_rel_sorted_index = sorted_similarity_index[target_rel - 1]
                    #     resorted_memory_pool = resort_memory(memory_pool, target_rel_sorted_index)


                    # MLLRE的做法
                    all_seen_data = []
                    for one_batch_memory in memory_data:
                        all_seen_data += one_batch_memory

                    memory_batch = memory_data[memory_index]
                    batch_train_data.extend(memory_batch)
                    # scores, loss = feed_samples(inner_model, memory_batch, loss_function, relation_numbers, device)
                    # optimizer.step()
                    memory_index = (memory_index+1) % len(memory_data)


                # random.shuffle(batch_train_data)
                if len(rel2instance_memory) > 0:  # from the second task, this will not be empty
                    if opt.is_curriculum_train == 'Y':
                        current_train_rel = batch_train_data[0][0]
                        current_rel_similarity_sorted_index = sorted_similarity_index[current_train_rel + 1]
                        seen_relation_sorted_index = []
                        for rel in current_rel_similarity_sorted_index:
                            if rel in rel2instance_memory.keys():
                                seen_relation_sorted_index.append(rel)

                        curriculum_rel_list = []
                        if opt.sampled_rel_num >= len(seen_relation_sorted_index):
                            curriculum_rel_list = seen_relation_sorted_index[:]
                        else:
                            step = len(seen_relation_sorted_index) // opt.sampled_rel_num
                            for i in range(0, len(seen_relation_sorted_index), step):
                                curriculum_rel_list.append(seen_relation_sorted_index[i])

                        # curriculum select relation
                        instance_list = []
                        for sampled_relation in curriculum_rel_list:
                            if opt.mini_batch_split == 'Y':
                                instance_list.append(rel2instance_memory[sampled_relation])
                            else:
                                instance_list.extend(rel2instance_memory[sampled_relation])
                    else:
                        # randomly select relation
                        instance_list = []
                        random_relation_list = random.sample(list(rel2instance_memory.keys()), min(opt.sampled_rel_num, len(rel2instance_memory)))
                        for sampled_relation in random_relation_list:
                            if opt.mini_batch_split == 'Y':
                                instance_list.append(rel2instance_memory[sampled_relation])
                            else:
                                instance_list.extend(rel2instance_memory[sampled_relation])

                    if opt.mini_batch_split == 'Y':
                        for one_batch_instance in instance_list:
                            # curriculum_instance_list = remove_unseen_relation(curriculum_instance_list, seen_relations)
                            scores, loss = feed_samples(inner_model, one_batch_instance, loss_function, relation_numbers, device)
                            optimizer.step()
                    else:
                        # curriculum_instance_list = remove_unseen_relation(curriculum_instance_list, seen_relations)
                        scores, loss = feed_samples(inner_model, instance_list, loss_function, relation_numbers, device)
                        optimizer.step()

                scores, loss = feed_samples(inner_model, batch_train_data, loss_function, relation_numbers, device)
                optimizer.step()
                total_loss += loss

            # valid test
            valid_acc = evaluate_model(inner_model, current_valid_data, opt.batch_size, relation_numbers, device)
            # checkpoint
            checkpoint = {'net_state': inner_model.state_dict(), 'optimizer': optimizer.state_dict()}
            if valid_acc > best_valid_acc:
                best_checkpoint = '%s/checkpoint_task%d_epoch%d.pth.tar' % (checkpoint_dir, task_ix + 1, epoch)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(checkpoint, best_checkpoint)
                best_valid_acc = valid_acc
                early_stop = 0
            else:
                early_stop += 1

            # print()
            t.set_description('Task %i Epoch %i' % (task_ix+1, epoch+1))
            t.set_postfix(loss=total_loss.item(), valid_acc=valid_acc, early_stop=early_stop, best_checkpoint=best_checkpoint)
            t.update(1)

            if early_stop >= opt.early_stop and task_ix != 0:
                # 已经充分训练了
                break

            if task_ix == 0 and early_stop >= 40:  # 防止大数据量的task在第一轮得不到充分训练
                break
        t.close()
        print('Load best check point from %s' % best_checkpoint)
        checkpoint = torch.load(best_checkpoint)

        weights_after = checkpoint['net_state']

        # weights_after = inner_model.state_dict()  # 经过inner_epoch轮次的梯度更新后weights
        # outer_step_size = opt.step_size * (1 - task_index / opt.task_num)  # linear schedule
        # if outer_step_size < opt.step_size * 0.5:
        #     outer_step_size = opt.step_size * 0.5

        if opt.outer_step_formula == 'fixed':
            outer_step_size = opt.step_size
        elif opt.outer_step_formula == 'linear':
            outer_step_size = opt.step_size * (1 - task_ix / opt.task_num)
        elif opt.outer_step_formula == 'square_root':
            outer_step_size = math.sqrt(opt.step_size * (1 - task_ix / opt.task_num))
        # outer_step_size = 0.4
        inner_model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outer_step_size
                                     for name in weights_before})

        # 用memory进行训练：
        # for i in range(5):
        #     for one_batch_memory in memory_data:
        #         scores, loss = feed_samples(inner_model, one_batch_memory, loss_function, relation_numbers, device)
        #         optimizer.step()


        results = [evaluate_model(inner_model, test_data, opt.batch_size, relation_numbers, device)
                   for test_data in current_test_data]  # 使用current model和alignment model对test data进行一个预测

        # sample memory from current_train_data
        if opt.memory_select_method == 'select_for_relation':
            # 每个关系sample k个
            for rel in train_relations:
                rel_items = remove_unseen_relation(train_data_dict[rel], seen_relations)
                rel_memo = select_data(inner_model, rel_items, int(opt.sampled_instance_num),
                                       relation_numbers, opt.batch_size, device)
                rel2instance_memory[rel] = rel_memo

        if opt.memory_select_method == 'select_for_task':
            # 为每个task sample k个
            rel_instance_num = math.ceil(opt.sampled_instance_num_total / len(train_relations))
            for rel in train_relations:
                rel_items = remove_unseen_relation(train_data_dict[rel], seen_relations)
                rel_memo = select_data(inner_model, rel_items, rel_instance_num,
                                       relation_numbers, opt.batch_size, device)
                rel2instance_memory[rel] = rel_memo

        if opt.task_memory_size > 0:
            # sample memory from current_train_data
            if opt.memory_select_method == 'random':
                memory_data.append(random_select_data(current_train_data, int(opt.task_memory_size)))
            elif opt.memory_select_method == 'vec_cluster':
                selected_memo = select_data(inner_model, current_train_data, int(opt.task_memory_size),
                                            relation_numbers, opt.batch_size, device)
                memory_data.append(selected_memo)  # memorydata是一个list，list中的每个元素都是一个包含selected_num个sample的list
                memory_pool.extend(selected_memo)
            elif opt.memory_select_method == 'difficulty':
                memory_data.append()

        print_list(results)
        avg_result = sum(results) / len(results)
        test_set_size = [len(testdata) for testdata in current_test_data]
        whole_result = sum([results[i] * test_set_size[i] for i in range(len(current_test_data))]) / sum(test_set_size)
        print('test_set_size: [%s]' % ', '.join([str(size) for size in test_set_size]))
        print('avg_acc: %.3f, whole_acc: %.3f' % (avg_result, whole_result))

        # end of each task, get embeddings of all
        if len(all_seen_relations) > 1:
            rel_embed = tsne_relations(inner_model, seen_task_relations, relation_numbers, device, task_sequence[opt.sequence_index])
            rel_embeddings.append(rel_embed)


    print('test_all:')
    for epoch in range(10):
        current_test_data = []
        for previous_task_id in range(opt.task_num):
            current_test_data.append(remove_unseen_relation(split_test_data[previous_task_id], seen_relations))

        loss_function = nn.MarginRankingLoss(opt.loss_margin)
        optimizer = optim.Adam(inner_model.parameters(), lr=opt.learning_rate)
        optimizer.zero_grad()
        for one_batch_memory in memory_data:
            scores, loss = feed_samples(inner_model, one_batch_memory, loss_function, relation_numbers, device)
            optimizer.step()
        results = [evaluate_model(inner_model, test_data, opt.batch_size, relation_numbers, device)
                   for test_data in current_test_data]
        print(results)
        avg_result = sum(results) / len(results)
        test_set_size = [len(testdata) for testdata in current_test_data]
        whole_result = sum([results[i] * test_set_size[i] for i in range(len(current_test_data))]) / sum(test_set_size)
        print('test_set_size: [%s]' % ', '.join([str(size) for size in test_set_size]))
        print('avg_acc: %.3f, whole_acc: %.3f' % (avg_result, whole_result))

    with open('./results/mllre_rel_embeddings_offset8.pkl', 'wb') as f:
        dump_tuple = (rel_embeddings, seen_task_relations, task_sequence[opt.sequence_index])
        pickle.dump(dump_tuple, f)





    # if opt.meta_method == 'reptile':
    #     # use reptile to train model
    #
    # elif opt.meta_method == 'maml':
    #     # use reptile to train model, wait implement
    #     pass
    # else:
    #     raise Exception('meta method %s not implement' % opt.meta_method)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', default=0, type=int,
                        help='cuda device index, -1 means use cpu')
    parser.add_argument('--train_file', default='dataset/training_data_with_entity.txt',
                        help='train file')
    parser.add_argument('--valid_file', default='dataset/val_data_with_entity.txt',
                        help='valid file')
    parser.add_argument('--test_file', default='dataset/val_data_with_entity.txt',
                        help='test file')
    parser.add_argument('--relation_file', default='dataset/relation_name.txt',
                        help='relation name file')
    parser.add_argument('--glove_file', default='dataset/glove.6B.300d.txt',
                        help='glove embedding file')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='word embeddings dimensional')
    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='BiLSTM hidden dimensional')
    parser.add_argument('--task_arrange', default='origin',
                        help='task arrangement method, e.g. origin, cluster_by_glove_embedding, random')
    parser.add_argument('--rel_encode', default='glove',
                        help='relation encode method')
    parser.add_argument('--meta_method', default='reptile',
                        help='meta learning method, maml and reptile can be choose')
    parser.add_argument('--num_cands', default=10, type=int,
                        help='candidate negative relation numbers in memory')
    parser.add_argument('--batch_size', default=50, type=float,
                        help='Reptile inner loop batch size')
    parser.add_argument('--task_num', default=10, type=int,
                        help='number of tasks')
    parser.add_argument('--train_instance_num', default=100, type=int,
                        help='number of instances for one relation, -1 means all.')
    parser.add_argument('--loss_margin', default=0.5, type=float,
                        help='loss margin setting')
    parser.add_argument('--outside_epoch', default=300, type=float,
                        help='task level epoch')
    parser.add_argument('--early_stop', default=20, type=float,
                        help='task level epoch')
    parser.add_argument('--step_size', default=0.4, type=float,
                        help='step size Epsilon')
    parser.add_argument('--outer_step_formula', default='fixed', type=str,
                        help='outer step formula, fixed, linear, square_root')
    parser.add_argument('--learning_rate', default=2e-3, type=float,
                        help='learning rate')
    parser.add_argument('--random_seed', default=100, type=int,
                        help='random seed')
    parser.add_argument('--task_memory_size', default=50, type=int,
                        help='number of samples for each task')
    parser.add_argument('--memory_select_method', default='select_for_relation',
                        help='the method of sample memory data, e.g. vec_cluster, random, difficulty, select_for_relation, select_for_task')
    parser.add_argument('--is_curriculum_train', default='Y',
                        help='when training with memory, this will control if relations are curriculumly sampled.')
    parser.add_argument('--mini_batch_split', default='N',
                        help='whether mini-batch split into sampled_rel_num batches, Y or N')
    parser.add_argument('--checkpoint_dir', default='./checkpoint',
                        help='check point dir')
    parser.add_argument('--sampled_rel_num', default=10,
                        help='relation sampled number for current training relation')
    parser.add_argument('--sampled_instance_num', default=6,
                        help='instance sampled number for a sampled relation, total sampled 6 * 80 instances ')
    parser.add_argument('--sampled_instance_num_total', default=50,
                        help='instance sampled number for a task, total sampled 50 instances ')
    parser.add_argument('--kl_dist_file', default='dataset/kl_dist_ht.json',
                        help='glove embedding file')
    parser.add_argument('--index', default=1, type=int,
                        help='experiment index')
    parser.add_argument('--sequence_index', default=8, type=int,
                        help='sequence index of tasks')

    opt = parser.parse_args()
    if opt.index == 1:
        # MLLRE
        opt.memory_select_method = 'vec_cluster'
        opt.task_memory_size = 50

    if opt.index == 2:
        opt.memory_select_method = 'select_for_task'
        opt.is_curriculum_train = 'N'
        opt.mini_batch_split = 'N'
        opt.checkpoint_dir = './checkpoint/2'
    #
    if opt.index == 3:
        opt.memory_select_method = 'select_for_task'
        opt.is_curriculum_train = 'N'
        opt.mini_batch_split = 'Y'
        opt.checkpoint_dir = './checkpoint/3'
    #
    if opt.index == 4:
        opt.memory_select_method = 'select_for_task'
        opt.is_curriculum_train = 'Y'
        opt.mini_batch_split = 'N'
        opt.checkpoint_dir = './checkpoint/4'
    #
    if opt.index == 5:
        opt.memory_select_method = 'select_for_task'
        opt.is_curriculum_train = 'Y'
        opt.mini_batch_split = 'Y'
        opt.checkpoint_dir = './checkpoint/5'
    #
    if opt.index == 6:
        opt.cuda_id = 1
        opt.memory_select_method = 'select_for_relation'
        opt.is_curriculum_train = 'N'
        opt.mini_batch_split = 'N'
        opt.checkpoint_dir = './checkpoint/6'
    #
    if opt.index == 7:
        opt.memory_select_method = 'select_for_relation'
        opt.is_curriculum_train = 'N'
        opt.mini_batch_split = 'Y'
        opt.checkpoint_dir = './checkpoint/7'
    #
    if opt.index == 8:
        opt.memory_select_method = 'select_for_relation'
        opt.is_curriculum_train = 'Y'
        opt.mini_batch_split = 'N'
        opt.checkpoint_dir = './checkpoint/8'
    #
    if opt.index == 9:
        opt.memory_select_method = 'select_for_relation'
        opt.is_curriculum_train = 'Y'
        opt.mini_batch_split = 'Y'
        opt.checkpoint_dir = './checkpoint/9'

    main(opt)