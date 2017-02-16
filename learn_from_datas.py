import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import os
import csv

######################################################
#######     VARIABLES
######################################################
TMP_FOLDER = os.getcwd() + "\\tmp\\"

DATAS_TRAIN_FILE = TMP_FOLDER + "datas_train.csv"
X_TRAIN_FILE = TMP_FOLDER + "X_train.csv"
Y_TRAIN_FILE = TMP_FOLDER + "Y_train.csv"

DATAS_TEST_FILE = TMP_FOLDER + "datas_test.csv"
X_TEST_FILE = TMP_FOLDER + "X_test.csv"
Y_TEST_FILE = TMP_FOLDER + "Y_test.csv"

VERBOSE = True
BATCH_SIZE = 3 # The number of training examples to use per training step.
NUM_EPOCHS = 10000
SVMC = 1
PLOT = True

######################################################
#######     FUNCTION DEFINITION
######################################################
def read_csv_file(path):
    #file = open(path, "r",encoding="utf8")
    tab = []
    with open(path, 'rU',encoding="utf8") as csvIN:
        outCSV = (line for line in csv.reader(csvIN, dialect='excel'))

        for row in outCSV:
            tab.append(row)
    return tab

######################################################
#######     LOAD DATAS
######################################################
datas_train = genfromtxt(DATAS_TRAIN_FILE, delimiter=";")
datas_train_size, num_features = np.array(datas_train).shape
num_features-=1
datas_train = datas_train[:num_features]

print("read datas_train : " + str(datas_train))

X_train = genfromtxt(X_TRAIN_FILE, delimiter=";")
Y_train = genfromtxt(Y_TRAIN_FILE, delimiter=";")
Y_train = np.transpose([Y_train])

# print("datas_train_size = " + str(datas_train_size) + " num_featues = " + str(num_features) + " Y_train " + str(Y_train))
# input()

datas_test = genfromtxt(DATAS_TEST_FILE, delimiter=";")
datas_test = datas_test[:num_features]
X_test = genfromtxt(X_TEST_FILE, delimiter=";")
Y_test = genfromtxt(Y_TEST_FILE, delimiter=";")
Y_test = np.transpose([Y_test])


######################################################
#######     TENSORFLOW LEARNING
######################################################
# Create graph
sess = tf.Session()

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
x = tf.placeholder("float", shape=[None, num_features], name="x")
y = tf.placeholder("float", shape=[None, 1], name="y")

# Define and initialize the network.

# These are the weights that inform how much each feature contributes to
# the classification.
W = tf.Variable(tf.zeros([num_features, 1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
y_raw = tf.matmul(x, W) + b

# Optimization.
regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]),1 - y * y_raw));
svm_loss = regularization_loss + SVMC * hinge_loss;
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

# Evaluation.
predicted_class = tf.sign(y_raw);
correct_prediction = tf.equal(y, predicted_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

if VERBOSE:
    print('Initialized!')
    print('Training.')

# Iterate and train.
for step in range(NUM_EPOCHS * datas_train_size // BATCH_SIZE):
# for step in range(6):
    if VERBOSE:
        print(step)

    offset = (step * BATCH_SIZE) % datas_train_size
    batch_data = X_train[offset:(offset + BATCH_SIZE), :]
    batch_labels = Y_train[offset:(offset + BATCH_SIZE)]
    print("Shapes batch_data = " + str(np.array(batch_data).shape) + " batch_labels = " + str(np.array(batch_labels).shape))
    sess.run(train_step,feed_dict={x: batch_data, y: batch_labels})
    print('loss: ', sess.run(svm_loss,feed_dict={x: batch_data, y: batch_labels}))

    if VERBOSE and offset >= datas_train_size - BATCH_SIZE:
        print

# Give very detailed output.
if VERBOSE:
    print('Weight matrix.')
    print(sess.run(W))
    print('Bias vector.')
    print(sess.run(b))
    print("Applying model to first test instance.")
print("Shapes X_train = " + str(np.array(X_train).shape) + " Y_train = " + str(np.array(Y_train).shape))
input()
print("Accuracy on train:", sess.run(accuracy,feed_dict={x: X_train, y: Y_train}))

# input()
# if PLOT:
#     eval_fun = lambda X: sess.run(predicted_class,feed_dict={x: X});
#     print("eval_fun")
#     input()
#     print("about to plot_boundary_on_data shapes : X_train = " + str(np.array(X_train).shape) + " , Y_train = " + str(np.array(Y_train).shape) + ", eval_fun = " + str(np.array(eval_fun).shape))
#     plot_boundary_on_data.plot(X_train, Y_train, eval_fun)
#     print("plot_boundary_on_data")
#     input()
W_final = sess.run(W)
b_final = sess.run(b)

print("Outputs shapes : W_final = " + str(np.array(W_final).shape) + " , b_final = " + str(np.array(b_final).shape))

# Create Saver
saver = tf.train.Saver()
# Save variables to .ckpt file
save_path = saver.save(sess, os.path.join(os.getcwd(), 'spam_detection_datas.ckpt'))
print("Model saved in file: %s" % save_path)