# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/13 20:58
# @author  : Mo
# @function: flask of model


# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
# 地址
from keras_textclassification.conf.path_config import path_model, path_fineture, path_model_dir, path_hyper_parameters
# 训练验证数据地址
from keras_textclassification.conf.path_config import path_baidu_qa_2019_train, path_baidu_qa_2019_valid
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessText, read_and_process, load_json
# 模型图
from keras_textclassification.m02_TextCNN.graph import TextCNNGraph as Graph
import numpy as np
# flask
from flask import Flask, request, jsonify
app = Flask(__name__)


hyper_parameters = load_json(path_hyper_parameters)
pt = PreprocessText(path_model_dir)
# 模式初始化和加载
graph = Graph(hyper_parameters)
graph.load_model()
ra_ed = graph.word_embedding
ques = '我要打王者荣耀'
# str to token
ques_embed = ra_ed.sentence2idx(ques)
if hyper_parameters['embedding_type'] in ['bert', 'albert']:
    x_val_1 = np.array([ques_embed[0]])
    x_val_2 = np.array([ques_embed[1]])
    x_val = [x_val_1, x_val_2]
else:
    x_val = ques_embed
# 预测
pred = graph.predict(x_val)
# 取id to label and pred
pre = pt.prereocess_idx(pred[0])


@app.route("/nlp/textcnn/predict", methods=["GET","POST"])
def predict():
    ques = request.args.get("text", "")
    ques_embed = ra_ed.sentence2idx(ques)
    if hyper_parameters['embedding_type'] in ['bert', 'albert']:
        x_val_1 = np.array([ques_embed[0]])
        x_val_2 = np.array([ques_embed[1]])
        x_val = [x_val_1, x_val_2]
    else:
        x_val = ques_embed
    pred = graph.predict(x_val)
    pre = pt.prereocess_idx(pred[0])
    pres_json = [dict(p) for p in pre]
    return jsonify(content_type="application/json;charset=utf-8",
                   reason="success",
                   charset="utf-8",
                   status="200",
                   content=pres_json)


if __name__ == "__main__":
    app.run(host="0.0.0.0",
            threaded=True,
            debug=True,
            port=8081)


# http://localhost:8081/nlp/textcnn/predict?text=你是谁



