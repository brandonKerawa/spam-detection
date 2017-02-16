import tensorflow as tf
from tensorflow.python.training import training as train
from nltk.corpus import stopwords #import stopwords from nltk corpus
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import FrenchStemmer
import numpy as np
from numpy import genfromtxt
import re
import os
import csv

######################################################
#######     VARIABLES
######################################################
TMP_FOLDER = os.getcwd() + "\\tmp\\"
TOKEN_DICTIONARY_FILE = TMP_FOLDER + "token_dictionary.csv"

COMPUTE_DATAS_FOLDER = os.getcwd() + "\\compute-datas\\"
# SPAM_RELATIVE_PATH = COMPUTE_DATAS_FOLDER + "Export_Annonces_SPAMs.csv"
# NON_SPAM_RELATIVE_PATH = COMPUTE_DATAS_FOLDER + "Export_Annonces_Non_Spams.csv"
# TEST_DATA_RELATIVE_PATH = COMPUTE_DATAS_FOLDER + "Data_Test.csv"
TEST_DATA_NON_SPAM_RELATIVE_PATH = COMPUTE_DATAS_FOLDER + "Data_Test_Non_SPAM.csv"

tokenDictionnary = genfromtxt(TOKEN_DICTIONARY_FILE, delimiter=";")

def read_csv_file(path):
    #file = open(path, "r",encoding="utf8")
    tab = []
    with open(path, 'rU',encoding="utf8") as csvIN:
        outCSV = (line for line in csv.reader(csvIN, dialect='excel'))

        for row in outCSV:
            tab.append(row)
    return tab

def convert_csv_to_computing_input(filepath):
    learningInput = []
    rawDatas = read_csv_file(filepath)
    i = 0
    for rawData in rawDatas:
        id = rawData[0]
        chaine = ''.join(rawData[1:])
        rawData[0] = chaine

        ##  Convert csv row to table
        rowTmp = rawData[0].split(";")

        # Extract title and description
        title = rowTmp[0]
        description = title + ' ' + ''.join(rowTmp[1:])

        # Create the row table
        row = [id,title,description]

        # Add the label column, by default set to 0
        # row.append(0)
        #  Convert to array
        #row = np.array(row, dtype=object)
        # print(str(len(row)) + "----" + str(row))
        learningInput.append(row)
    return learningInput #np.array(learningInput)

def convert_computing_input_to_dictionnary_input(datas):
    # Instantiation of the Tokenizer
    tokenizer = WordPunctTokenizer()
    # Instantiate Stemmer
    stemmer = FrenchStemmer()
    # Load french StopWords
    french_stopwords = set(stopwords.words('french'))

    matchingTable = []
    tokenTable = [] # Each row of this table is [id, tokens] :'id' of the advert and 'tokens' the list of tokens in the advert
    i = 0
    for row in datas:
        id = row[0]
        desc = row[2]
        # Get tokens for this row
        tokens = tokenizer.tokenize(str(desc[0]))
        # Filter tokens to remove punctuation
        regex = re.compile(r'\w+')
        tokens = filter(regex.search, tokens)
        # Filter tokens to remove stopwords and convert tokens to their stemm
        tokens = [stemmer.stem(token) for token in tokens if token.lower() not in french_stopwords]
        # Remove duplicate entries
        tokens = list(set(tokens))
        # Sort tokens
        tokens.sort()
        # Construct the new row with only the id and the list of tokens
        row = [id,tokens]

        # Add the new Row to the global table
        tokenTable.append(row)
        i += 1

    # Construct the vector for each advert
    rowCol = 0
    rowCols = []
    tabCols = []
    rowInd = 0
    initialRow = np.zeros(len(tokenDictionnary))


    #
    # # Here we transform each row of tokens into row of 0|1 corresponding array, matching the tokenDictionnary
    #
    # tokenTable[1:] to skip the title row, because the original file has a title row
    for row in tokenTable[1:]:
        id = row[0].split(";")[0]
        advertVec = np.zeros(len(tokenDictionnary))
        rowCols = []
        for elm in row[1]:
            rowCol = tokenDictionnary.index(elm) if elm in tokenDictionnary else -1
            advertVec[rowCol] = 1 if rowCol >= 0 else 0

        composed_row = [id,advertVec]
        matchingTable.append(composed_row)
        rowInd += 1
    return tokenTable, matchingTable

datas = convert_csv_to_computing_input(TEST_DATA_NON_SPAM_RELATIVE_PATH)
print("Shape datas = " + str(np.array(datas).shape))
# input()
print("datas = " + str(datas))
# input()

tokenTable, matchingTable = convert_computing_input_to_dictionnary_input(datas)
# print("Shape tokenTable = " + str(np.array(tokenTable).shape) + " matchingTable = " + str(np.array(matchingTable).shape))
# input()
# print("tokenTable = " + str(tokenTable))
# input()
# print("matchingTable = " + str(matchingTable))
# input()

# Create graph
sess = tf.Session()

# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# Restore variables from disk.
# saver.restore(sess, os.path.join(os.getcwd(), 'spam_detection_datas.ckpt'))
variables = train.NewCheckpointReader(os.path.join(os.getcwd(), 'spam_detection_datas.ckpt'))
print("Model restored.")

W = variables.get_tensor("W")
b = variables.get_tensor("b")
W = np.transpose(W)
print("W" + str(W) + "\n b" + str(b))

# print("Shapes W" + str(np.array(W).shape) + "\n X" + str(np.array(np.transpose([matchingTable[0][1]])).shape) + " b = " + str(np.array(b).shape))
input()
# Process Datas
output = []
for row in matchingTable:
    id = row[0]
    X = row[1]
    # X = np.transpose([row[1]])
    # X = np.transpose(X)
    # [W] = np.transpose(W)
    print("Shapes W" + str(np.array(W).shape) + "\n X" + str(np.array(X).shape))
    Y = np.matmul(W,X)+b
    y = np.sign(Y[0])
    output.append([id,y])

print("output = " + str(output))
np.savetxt(COMPUTE_DATAS_FOLDER+'output.csv', np.array(output), fmt = ["%s"] + ["%.3f",],delimiter = ';')
print("output.csv created Shape = " + str(np.array(output).shape))
