import os

import json

import cv2
import numpy as np


import traceback

GENDR_ID2NMAE = {0: "male", 1: "female", 2: "unknown"}


class FacialDB(object):
    '''
    人脸信息数据库
    Facial info database
    存储人脸对应的姓名、性别、特征向量等，可执行查询方法
    '''
    _DB_DICT = {}

    def __init__(self, path_json=None, db_dict=None) -> None:
        if not path_json is None:
            self.load_json(path_json)
        if not db_dict is None:
            self.update(db_dict)

    def query_N2N(self, features_tocheck, known_features=None, threshold=0.6):
        '''
        多对多的人脸特征向量查询
        features_tocheck: 待查询的特征向量，shape=(N, 512)
        known_features: 已知的特征向量，shape=(512, M)
        threshold: 阈值，大于该阈值则认为是同一个人
        '''
        features_tocheck = np.ascontiguousarray(features_tocheck)
        known_features = (
            self.known_mean_features if known_features is None else known_features)

        dists_N = np.dot(features_tocheck, known_features)

        dists_max = dists_N.max(axis=-1)
        inds = dists_N.argmax(axis=-1)
        knowns = dists_max > threshold

        return inds, knowns, dists_max

    def query(self, feature_tocheck, known_features=None, threshold=0.6):
        inds, knowns, dists_max = self.query_N2N(
            [feature_tocheck], known_features, threshold)
        return inds[0], knowns[0], dists_max[0]

    @ property
    def db_dict(self):
        return self._DB_DICT.copy()

    @property
    def known_mean_features(self):
        list_features = [d["feature_vector"] for d in self._DB_DICT.values()]
        return np.ascontiguousarray(list_features).T

    @property
    def index2id(self):
        return {ind: id_person for ind, id_person in enumerate(self._DB_DICT.keys())}

    @property
    def id2name(self):
        return {k: v["name"] for k, v in self._DB_DICT.items()}

    @property
    def ind2name(self):
        return {ind: v["name"] for ind, v in enumerate(self._DB_DICT.values())}

    @property
    def all_names(self):
        return [v["name"] for v in self._DB_DICT.values()]

    @property
    def nb_people(self):
        return len(self._DB_DICT.keys())

    def append(self, id, info_dict):
        self._DB_DICT.update({id: info_dict})

    def update(self, db_dict):
        self._DB_DICT.update(db_dict)

    def load_json(self, path_json):
        with open(path_json, "r") as fp:
            db_dict = json.load(fp)
        self.update(db_dict)

    def save_to_json(self, path_json):
        with open(path_json, "w") as fp:
            json.dump(self.db_dict)




def _check_keys(func):
    def inner(*args):
        self = args[0]
        key = args[1]
        if not key in self._INFO_DICT.keys():
            raise KeyError(
                "Info key : {} is not aviliable in FacialInfo".format(key))
        return func(*args)
    return inner


class FacialInfo(object):
    _INFO_DICT = {"name": None, "id": None,
                  "gender_id": None, "feature_vector": None, 'feature_list': None}

    def __init__(self, path_json=None, info_dict=None, ) -> None:
        if not path_json is None:
            self.load_info(path_json)
        elif not info_dict is None:
            self.update(info_dict)

    @property
    def info_dict(self):
        return self._INFO_DICT.copy()

    @_check_keys
    def set(self, key, value):

        self._INFO_DICT.update({key: value})

    @_check_keys
    def get(self, key):
        return self._INFO_DICT.get(key, None)

    def update(self, info_dict):
        self._INFO_DICT.update(info_dict)

    def load_info(self, path_json):
        with open(path_json, 'r') as fp:
            info_dict = json.load(fp)

        for key in info_dict.keys():
            if not key in self._INFO_DICT.keys():
                raise KeyError(
                    "Info key : {} is not aviliable in FacialInfo".format(key))

        self._INFO_DICT.update(info_dict)

        return self._INFO_DICT

    def save_info(self, path_json):
        try:
            with open(path_json, 'w') as fp:
                json.dump(self._INFO_DICT, fp)
        except:
            traceback.print_exc()
            os.remove(path_json)


def parse_filename(path_dir):  # path format: xxx_id_gender
    name_dir = os.path.split(path_dir)[-1]

    name_celeba, id_douban, gender_char = name_dir.split('_')
    gender_id = {'m': 0, 'f': 1, 'u': 2}[gender_char]
    if id_douban.lower() == "n":
        id_douban
    return name_celeba, id_douban, gender_id


def remove_old(root_dir):
    pathes_dir = [path_dir for path_dir in [os.path.join(
        root_dir, name_dir) for name_dir in os.listdir(root_dir)] if os.path.isdir(path_dir)]
    for path_dir in pathes_dir:
        info_path = os.path.join(path_dir, 'info.json')
        if os.path.exists(info_path):
            os.remove(info_path)

def imread(filename):
    return cv2.imdecode(np.fromfile(file=filename, dtype=np.uint8), cv2.IMREAD_COLOR)
