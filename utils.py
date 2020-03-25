import pickle as pkl
import json


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        vec = pkl.load(f)
        return vec


def dump_pickle(file_path, obj):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def dump_json(file_path, obj):
    with open(file_path, 'w') as f:
        json.dump(obj, f)


def remove_unseen_relation(dataset, seen_relations):
    cleaned_data = []
    for data in dataset:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            cleaned_data.append([data[0], data[1][-2:], data[2]])
    return cleaned_data