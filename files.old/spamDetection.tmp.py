######################################################
#######     LIBRARY IMPORT
######################################################

import nltk #import the natural language toolkit library
from nltk.stem.snowball import FrenchStemmer #import the French stemming library
from nltk.corpus import stopwords #import stopwords from nltk corpus
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import FrenchStemmer
import re #import the regular expressions library; will be used to strip punctuation
import numpy as np
from collections import Counter #allows for counting the number of occurences in a list
import os #import os module
import csv
import tensorflow as tf
import random
from matplotlib import pyplot as plt
import plot_boundary_on_data

######################################################
#######     VARIABLES
######################################################
SPAM_RELATIVE_PATH = "\\data\\Export_Annonces_SPAMs.csv"
NON_SPAM_RELATIVE_PATH = "\\data\\Export_Annonces_Non_Spams.csv"
TEST_DATA_RELATIVE_PATH = "\\data\\Data_Test.csv"
TEST_DATA_NON_SPAM_RELATIVE_PATH = "\\data\\Data_Test_Non_SPAM.csv"

SPAM_VALUE = 1
NON_SPAM_VALUE = -1

VERBOSE = True
BATCH_SIZE = 100  # The number of training examples to use per training step.
NUM_EPOCHS = 1000
SVMC = 1
PLOT = True

tokenDictionnary = []
# Instantiation of the Tokenizer
tokenizer = WordPunctTokenizer()
# Instantiate Stemmer
stemmer = FrenchStemmer()
# Load french StopWords
french_stopwords = set(stopwords.words('french'))

######################################################
#######     FUNCTION DEFINITION
######################################################
#reading in the raw text from the file
def read_raw_file(path):
    '''reads in raw text from a text file using the argument (path), which represents the path/to/file'''
    f = open(path,"r",encoding="utf8") #open the file located at "path" as a file object (f) that is readonly
    raw = f.read() # read raw text into a variable (raw) after decoding it from utf8
    f.close() #close the file now that it isn;t being used any longer
    return raw

def read_csv_file(path):
    #file = open(path, "r",encoding="utf8")
    tab = []
    with open(path, 'rU',encoding="utf8") as csvIN:
        outCSV = (line for line in csv.reader(csvIN, dialect='excel'))

        for row in outCSV:
            tab.append(row)
    return tab

def match_elm_dictionary(element,dictionary, defaultRow):
    """
    :param element: array containing strings.
    :param dictionary: dictionary containing all words
    :param defaultRow : default value of the row. It is better to have defaultRow = np.zeros(len(dictionary))
    :return: return an array of size len(dictionary) where the element at the index i is equal to zero if the ith word of the dictionary is into the array element
    """
    output = []
    rowCol = 0
    advertVec = defaultRow
    for elm in element:
        rowCol = dictionary.index(elm) if elm in dictionary else -1
        advertVec[rowCol] = 1 if rowCol >= 0 else 0
    output.append(advertVec)
    return output

def convert_csv_to_learning_input(filepath,yValue):
    learningInput = []
    rawSpams = read_csv_file(filepath)
    i = 0
    for rawRow in rawSpams:
        #print(str(len(rawRow)) + "----" "rawRow start " + str(rawRow))
        chaine = ''.join(rawRow)
        rawRow[0] = chaine

        ##  Convert csv row to table
        rowTmp = rawRow[0].split(";")

        # Extract title and description
        title = rowTmp[0]
        description = title + ' ' + ''.join(rowTmp[1:])

        # Create the row table
        row = [title,description]

        # Add y column to input data
        row.append(yValue)
        #  Convert to array
        #row = np.array(row, dtype=object)
        #print(str(len(row)) + "----" + str(row))
        learningInput.append(row)
    return learningInput #np.array(learningInput)

def preprocess_data(datas):
    tokenDictionnary=[]
    matchingTable = []
    tokenTable = []
    i=0
    for row in datas:
        row = row[1:]
        # Get tokens for this row
        tokens = tokenizer.tokenize(str(row[0]))
        # Filter tokens to remove punctuation
        regex = re.compile(r'\w+')
        tokens = filter(regex.search, tokens)
        # Filter tokens to remove stopwords and convert tokens to their stemm
        tokens = [stemmer.stem(token) for token in tokens if token.lower() not in french_stopwords]
        # Remove duplicate entries
        tokens = list(set(tokens))
        tokens.sort()
        row[0] = tokens

        # Add new Tokens to the Dictionnary
        tokenDictionnary.extend(tokens)
        tokenDictionnary = list(set(tokenDictionnary))
        tokenDictionnary.sort()

        # Add the new Row to the global table
        tokenTable.append(row)
        i += 1

    # Construct the vector for each advert
    rowCol = 0
    rowCols = []
    tabCols = []
    rowInd = 0
    initialRow = np.zeros(len(tokenDictionnary))
    for row in tokenTable[1:]:
        advertVec = np.zeros(len(tokenDictionnary)+1)
        rowCols = []
        for elm in row[0]:
            rowCol = tokenDictionnary.index(elm) if elm in tokenDictionnary else -1
            advertVec[rowCol] = 1 if rowCol >= 0 else 0
            # rowCols.append(rowCol)

        # print(str(rowInd) + " - rC - " + str(rowCols))
        # print( str(rowInd) + " - advertVec - " + str(len(row[0])) + " - "+ str(advertVec))
        advertVec[len(tokenDictionnary)] = row[len(row)-1]
        #advertVec.append(row[len(row) - 1])
        matchingTable.append(advertVec)
        rowInd += 1
    return tokenTable, matchingTable, tokenDictionnary

######################################################
#######     DATA PREPROCESSING
######################################################
tokenDictionnary = []
tokenTable = []
matchingTable = []

# Extract spams and non spams from files
rawSpams = convert_csv_to_learning_input(os.getcwd()+ TEST_DATA_RELATIVE_PATH,SPAM_VALUE)
rawNonSpams = convert_csv_to_learning_input(os.getcwd()+ TEST_DATA_NON_SPAM_RELATIVE_PATH,NON_SPAM_VALUE)

spams_size = len(rawSpams)
non_spams_size = len(rawNonSpams)

# Concatenate Spams and NonSpams into one array
rawDatas = rawSpams[1:] + rawNonSpams[1:]
#print(str(rawDatas))
#input()

# Build the dictionnary of words and the matching table
tokenTable, matchingTable, tokenDictionnary = preprocess_data(rawDatas)

# Extract processed datas spams and datas non spams
datasSpams = matchingTable[:spams_size]
datasNonSpams = matchingTable[spams_size:]

# Extract training datas
datas_size, num_features = np.array(matchingTable).shape
#print("shape = " + str(np.array(rawDatas).shape))
num_features-=1

train_spams_size = (2*spams_size)//3
test_spams_size = spams_size - train_spams_size

train_non_spams_size = (2*non_spams_size)//3
test_non_spams_size = non_spams_size - train_non_spams_size

datas_train = datasSpams[:train_spams_size] + datasNonSpams[:train_non_spams_size]
datas_train_size = train_spams_size + train_non_spams_size


X_train = np.array(datas_train)[:,:num_features]
Y_train = np.array(datas_train)[:,num_features:]

datas_test = datasSpams[train_spams_size:] + datasNonSpams[train_non_spams_size:]
X_test= np.array(datas_test)[:,:num_features]
Y_test= np.array(datas_test)[:,num_features:]

X_datas = np.concatenate((X_train, X_test),axis=0)
print(X_datas.shape)
Y_datas = np.concatenate((Y_train, Y_test),axis=0)

# print("############################### Y Train ############ ")
# print("############################### Datas Test ############ shape = " + str(np.array(datas_test).shape))
# input()
# i=0
# for row in X_train:
#     i+=1
#     print( str(i) + "----- " + " len " + str(row))
# input()
#
# i=0
# for row in datas_test:
#     i+=1
#     print( str(i) + "----- " + " len " + str(len(row)) + str(row) )
# input()
######################################################
#######     INITIALIZE TENSORFLOW SESSION
######################################################
# Create graph
sess = tf.Session()

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
x = tf.placeholder("float", shape=[None, num_features])
y = tf.placeholder("float", shape=[None,1])

# Define and initialize the network.

# These are the weights that inform how much each feature contributes to
# the classification.
W = tf.Variable(tf.zeros([num_features,1]))
b = tf.Variable(tf.zeros([1,1]))

# Declare model operations
model_output = tf.sub(tf.matmul(x, W), b)
y_raw = tf.matmul(x,W) + b

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(W))

# Declare loss function
# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# L2 regularization parameter, alpha
alpha = tf.constant([0.01])
# Margin term in loss
classification_term = tf.reduce_mean(tf.maximum(0., tf.sub(1., tf.mul(model_output, y))))
# Put terms together
loss = tf.add(classification_term, tf.mul(alpha, l2_norm))

# Declare prediction function
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(NUM_EPOCHS):
    rand_index = np.random.choice(len(X_train), size=BATCH_SIZE)
    rand_x = X_train[rand_index]
    rand_y = Y_train[rand_index]

    sess.run(train_step, feed_dict={x: rand_x, y: rand_y})

    temp_loss = sess.run(loss, feed_dict={x: rand_x, y: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={x: X_test, y: Y_test})
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 100 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(W)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# Extract coefficients
tmp = sess.run(W)
print("tmp" + str(tmp) + "shape " + str(tmp.shape))
input()
[[a1], [a2]] = tmp
[[b]] = sess.run(b)
slope = -a2 / a1
y_intercept = b / a1

# Extract x1 and x2 vals
x1_vals = [d[1] for d in X_datas]

# Get best fit line
best_fit = []
for i in x1_vals:
  best_fit.append(slope*i+y_intercept)

# Separate I. setosa
setosa_x = [d[1] for i,d in enumerate(X_datas) if Y_datas[i]==1]
setosa_y = [d[0] for i,d in enumerate(X_datas) if Y_datas[i]==1]
not_setosa_x = [d[1] for i,d in enumerate(X_datas) if Y_datas[i]==-1]
not_setosa_y = [d[0] for i,d in enumerate(X_datas) if Y_datas[i]==-1]

# Plot data and line
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot train/test accuracies
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

######################################################
#######     MINI TEST
######################################################
#nonSpams = read_csv_file(os.getcwd()+ NON_SPAM_RELATIVE_PATH)
# i=0
# # for row in tokenTable:
# #     i = i + 1
# #     if i%20 == 0:
# #         input()
# #     print(str(i) + " --- len --- " + str(len(row)) + "-------- " + str(row) )
#
# print("size tokenDictionnay" + str(len(tokenDictionnary)) + "\n" + str(tokenDictionnary[0:10]))
# l=0
#
# print("################################################\n#############################  TOKEN TABLE\n#############################################")
# for row in tokenTable:
#     l = l + 1
#     if l%50  == 0:
#         input()
#     print(str(l) + " --- len --- " + str(len(row[0])) + "-------- " + str(row))
# k=0
#
# print("################################################\n#############################  MATCHING TABLE \n#############################################")
# for row in matchingTable:
#     k = k + 1
#     print(str(k) + " --- len --- " + str(np.sum(row)) + "-------- " + str(row))


######################################################
#######     CONCLUSION
######################################################