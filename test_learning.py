import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from tensorflow.python.training import training as train
import os


TMP_FOLDER = os.getcwd() + "\\tmp\\"

DATAS_TRAIN_FILE = TMP_FOLDER + "datas_train.csv"

DATAS_TEST_FILE = TMP_FOLDER + "datas_test.csv"
X_TEST_FILE = TMP_FOLDER + "X_test.csv"
Y_TEST_FILE = TMP_FOLDER + "Y_test.csv"

######################################################
#######     LOAD TESTING DATAS
######################################################
datas_train = genfromtxt(DATAS_TRAIN_FILE, delimiter=";")
datas_train_size, num_features = np.array(datas_train).shape
num_features-=1
datas_train = datas_train[:num_features]

# print("datas_train_size = " + str(datas_train_size) + " num_featues = " + str(num_features) + " Y_train " + str(Y_train))
# input()

datas_test = genfromtxt(DATAS_TEST_FILE, delimiter=";")
datas_test = datas_test[:num_features]
X_test = genfromtxt(X_TEST_FILE, delimiter=";")
Y_test = genfromtxt(Y_TEST_FILE, delimiter=";")
Y_test = np.transpose([Y_test])

######################################################
#######     LOAD LEARNING VARIABLES
######################################################
# Create graph
sess = tf.Session()

# Load Learning Variables
variables = train.NewCheckpointReader(os.path.join(os.getcwd(), 'spam_detection_datas.ckpt'))
print("Model restored.")

# Get the Weight and Bias Vectors
W = variables.get_tensor("W")
b = variables.get_tensor("b")

print("X_test" + str(X_test) + "\n Y_test" + str(Y_test))
# Do some work with the model

######################################################
#######     TEST
######################################################

# Placeholder definition
x = tf.placeholder("float", shape=[None, num_features], name="x")
y = tf.placeholder("float", shape=[None, 1], name="y")
y_raw = tf.matmul(x, W) + b

# Define Accuracy Evaluation
predicted_class = tf.sign(y_raw);
correct_prediction = tf.equal(y, predicted_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Result
print("Shapes X_test = " + str(np.array(X_test).shape) + " Y_test = " + str(np.array(Y_test).shape))
accuracy_value = sess.run(accuracy,feed_dict={x: X_test, y: Y_test})
print("Accuracy = " + str(accuracy_value))