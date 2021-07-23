# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/13 15:41
# @author  : Mo
# @function:


import tensorflow as tf


def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
    receiver_tensors      = {"predictor_inputs": serialized_tf_example}
    feature_spec          = {"words": tf.FixedLenFeature([25],tf.int64)}
    features              = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def estimator_spec_for_softmax_classification(logits, labels, mode):
    predicted_classes = tf.argmax(logits, 1)
    if (mode == tf.estimator.ModeKeys.PREDICT):
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"pred_output_classes": predicted_classes, 'probabilities': tf.nn.softmax(logits)})}
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'class': predicted_classes, 'prob': tf.nn.softmax(logits)}, export_outputs=export_outputs) # IMPORTANT!!!

    onehot_labels = tf.one_hot(labels, 31, 1, 0)
    loss          = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    if (mode == tf.estimator.ModeKeys.TRAIN):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op  = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def model_custom(features, labels, mode):
    bow_column           = tf.feature_column.categorical_column_with_identity("words", num_buckets=1000)
    bow_embedding_column = tf.feature_column.embedding_column(bow_column, dimension=50)
    bow                  = tf.feature_column.input_layer(features, feature_columns=[bow_embedding_column])
    logits               = tf.layers.dense(bow, 31, activation=None)

    return estimator_spec_for_softmax_classification(logits=logits, labels=labels, mode=mode)


def main_():
    # ...
    # preprocess-> features_train_set and labels_train_set
    # ...
    classifier     = tf.estimator.Estimator(model_fn = model_custom)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"words": features_train_set}, y=labels_train_set, batch_size=batch_size_param, num_epochs=None, shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=100)

    full_model_dir = classifier.export_savedmodel(export_dir_base="C:/models/directory_base", serving_input_receiver_fn=serving_input_receiver_fn)


def main_tst():
    # ...
    # preprocess-> features_test_set
    # ...
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
        predictor   = tf.contrib.predictor.from_saved_model(full_model_dir)
        model_input = tf.train.Example(features=tf.train.Features( feature={"words": tf.train.Feature(int64_list=tf.train.Int64List(value=features_test_set)) }))
        model_input = model_input.SerializeToString()
        output_dict = predictor({"predictor_inputs":[model_input]})
        y_predicted = output_dict["pred_output_classes"][0]


