# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/18 21:02
# @author   :Mo
# @function :tools of file utils

import json


def save_json(jsons, json_path):
    """
        保存为json文件
    :param jsons: json, input of json
    :param json_path: str, file path of json
    :return: None
    """
    with open(json_path, 'w', encoding='utf-8') as fs:
        fs.write(json.dumps(jsons))
    fs.close()


def load_json(json_path):
    """
        下载json文件
    :param json_path:str, file path of json 
    :return: json
    """
    with open(json_path, 'r', encoding='utf-8') as fl:
        model_json = json.loads(fl.readlines()[0])
    fl.close()
    return model_json