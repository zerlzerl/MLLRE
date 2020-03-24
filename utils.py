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
