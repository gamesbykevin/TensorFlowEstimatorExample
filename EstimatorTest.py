from tensorflow.contrib.learn.python.learn.datasets import base

import tensorflow as tf
import numpy as np

# print tensor flow version
print('TF Version: ', tf.__version__)

# data file which we will train on
TRAIN = "candles_train.txt"

# data file which we will test to determine accuracy
TEST = "candles_test.txt"

# training set
train_set = base.load_csv_without_header(filename=TRAIN, features_dtype=np.double, target_dtype=np.double)

# test set
test_set = base.load_csv_without_header(filename=TEST, features_dtype=np.double, target_dtype=np.double)

# print train data set
# print(train_set.data)

# print test data set
# print(test_set.data)

# add feature columns so tensor flow will know what we need to train on
feature_name = "stock_data_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[1])]

# our classifier will do the training as well as keep track of the state if we need to use it again
classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3, model_dir="/tmp/model")

# connect model to training data
def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return _fn

print(input_fn(train_set))

# raw data -> input function -> feature columns -> model

# fit the model with our data set and train on 1,000 rows
classifier.train(input_fn=input_fn(train_set), steps=1000)

print('done')

# evaluate accuracy by testing 100 rows
accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), steps=100)["accuracy"]

# print our accuracy
print('\nAccuracy: {0:f}'.format(accuracy_score))