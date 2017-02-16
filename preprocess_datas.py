#!/usr/bin/env python
# coding: utf-8
######################################################
#######     LIBRARY IMPORT
######################################################

from nltk.corpus import stopwords #import stopwords from nltk corpus
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import FrenchStemmer
import re #import the regular expressions library; will be used to strip punctuation
import numpy as np
import os #import os module
import csv
import pickle
import time
import multiprocessing
import encodings
import warnings

######################################################
#######     VARIABLES
######################################################
INPUT_DATAS_FOLDER = os.getcwd() + "\\input-datas\\"
SPAM_RELATIVE_PATH = INPUT_DATAS_FOLDER + "Export_Annonces_SPAMs.csv"
NON_SPAM_RELATIVE_PATH = INPUT_DATAS_FOLDER + "Export_Annonces_Non_Spams.csv"
TEST_DATA_RELATIVE_PATH = INPUT_DATAS_FOLDER + "Data_Test.csv"
TEST_DATA_NON_SPAM_RELATIVE_PATH = INPUT_DATAS_FOLDER + "Data_Test_Non_SPAM.csv"

TMP_FOLDER = os.getcwd() + "\\tmp\\"

SPAM_VALUE = 1
NON_SPAM_VALUE = -1

tokenDictionnary = []



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
    with open(path, 'rU',encoding="unicode_escape") as csvIN:
        csv_reader = csv.reader(csvIN, dialect='excel')
        # print("csv_reader = " + str(csv_reader))
        for row in csv_reader:
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
    # print(str(filepath) + "shape = " + str(np.array(rawSpams).shape ))
    number_line_skipped = 0
    i = 0
    for rawRow in rawSpams:
        # print(str(len(rawRow)) + "----" "rawRow start " + str(rawRow))
        # print(str(i) + " ---- rawRow shape = " + str(np.array(rawRow).shape) + " Value = " + str(rawRow))
        # input()
        try:
            chaine = ''.join(rawRow)
            rawRow[0] = chaine

            ##  Convert csv row to table
            rowTmp = rawRow[0].split(";")

            # Extract title and description
            title = rowTmp[0]
            description = title + ' ' + ''.join(rowTmp[1:])
            title = title.encode('iso-8859-1')
            title = title.decode('utf8')

            description = description.encode('iso-8859-1')
            description = description.decode('utf8')
            # Create the row table
            row = [title,description]

            # Add y column to input data
            row.append(yValue)
            #  Convert to array
            #row = np.array(row, dtype=object)
            # print(str(i) + " ---- row type title = " + str(type(title)) + str(len(row)) + " ---- " + str(row))
            learningInput.append(row)
            # input()
        except:
            # print("line " + str(i) + " skipped")
            number_line_skipped +=1
            continue
        i+=1
    print(str(number_line_skipped) + " lines skipped while processing " + filepath)
    return learningInput #np.array(learningInput)

def build_dictionary_token_table(datas):
    # Instantiation of the Tokenizer
    tokenizer = WordPunctTokenizer()
    # Instantiate Stemmer
    stemmer = FrenchStemmer()
    # Load french StopWords
    french_stopwords = set(stopwords.words('french'))

    tokenDictionary=[]
    matchingTable = []
    tokenTable = []
    i=0
    print("\tBuilding the dictionary ...")
    start_time = time.time()
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
        tokenDictionary.extend(tokens)
        tokenDictionary = list(set(tokenDictionary))
        tokenDictionary.sort()

        # Add the new Row to the global table
        tokenTable.append(row)
        if i%1000 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            start_time = end_time
            print("\t\t" + str(i)+" rows processed - elapse : " + str(elapsed) + "...")
        i += 1
    return tokenTable, tokenDictionary

def match_token_list_with_dictionary(token_list,dictionary):
    advertVec = np.zeros(len(tokenDictionary) + 1)
    for elm in token_list:
        rowCol = tokenDictionary.index(elm) if elm in set(dictionary) else -1
        advertVec[rowCol] = 1 if rowCol >= 0 else 0
    return advertVec

def build_matching_table(tokenTable, tokenDictionary,matchingTable):
    # Construct the vector for each advert
    print("\tBuilding the matching vector for each row...")
    rowInd = 0
    start_time = time.time()
    i=0
    for row in tokenTable[1:]:
        advertVec = np.zeros(len(tokenDictionary)+1)
        for elm in row[0]:
            rowCol = tokenDictionary.index(elm) if elm in set(tokenDictionary) else -1
            advertVec[rowCol] = 1 if rowCol >= 0 else 0

        # print(str(rowInd) + " - rC - " + str(rowCols))
        # print( str(rowInd) + " - advertVec - " + str(len(row[0])) + " - "+ str(advertVec))
        advertVec[len(tokenDictionary)] = row[len(row)-1]
        #advertVec.append(row[len(row) - 1])
        matchingTable.append(advertVec)
        if rowInd%1000 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            start_time = end_time
            print("\t\t" + str(rowInd) + " rows processed - elapse : " + str(elapsed) + " --- rowCol was " + str(rowCol))
        rowInd += 1
    return matchingTable

######################################################
#######     DATA PREPROCESSING
######################################################
tokenDictionary = []
tokenTable = []
matchingTable = []

# # Extract spams and non spams from files
rawSpams = convert_csv_to_learning_input(SPAM_RELATIVE_PATH,SPAM_VALUE)
print(SPAM_RELATIVE_PATH + " file processed\n\n")
rawNonSpams = convert_csv_to_learning_input(NON_SPAM_RELATIVE_PATH,NON_SPAM_VALUE)
print(NON_SPAM_RELATIVE_PATH + " file processed\n\n")

# input()
spams_size = len(rawSpams)
non_spams_size = len(rawNonSpams)
#
# # Concatenate Spams and NonSpams into one array
# rawDatas = rawSpams[1:] + rawNonSpams[1:]
# #print(str(rawDatas))
# # input()
#
# # Build the dictionnary of words and the matching table
# print(" Building the tokens' dictionary and the matching table ...")
# tokenTable, tokenDictionary = build_dictionary_token_table(rawDatas)
# matchingTable = build_matching_table(tokenTable, tokenDictionary)
# # np.savetxt( TMP_FOLDER + "token_table.csv", np.array(tokenTable), delimiter=";",fmt="%s")
#
# try:
#     file = open(TMP_FOLDER + "token_table",'wb')
#     pickle.dump(tokenTable,file)
#     print("'token_table' created | Shape = " + str(np.array(tokenTable).shape))
#     file.close()
# except:
#     print("failed to create file 'token_table'")
#
# try:
#     file = open(TMP_FOLDER + "token_dictionary",'wb')
#     pickle.dump(tokenDictionary,file)
#     print("'token_dictionary' created | Shape = " + str(np.array(tokenDictionary).shape))
#     file.close()
# except:
#     print("failed to create file 'token_dictionary'")

file = open(TMP_FOLDER + "token_table",'rb')
tokenTable = pickle.load(file)
file.close()

file = open(TMP_FOLDER + "token_dictionary",'rb')
tokenDictionary = pickle.load(file)
file.close()

procs = 2   # Number of processes to create

# Create a list of jobs and then iterate through
# the number of processes appending each process to
# the job list
jobs = []
for i in range(0, procs):
    out_list = list()
    process = multiprocessing.Process(target=build_matching_table,
                                      args=(tokenTable, tokenDictionary,matchingTable))
    jobs.append(process)

# Start the processes (i.e. calculate the random number lists)
for j in jobs:
    j.start()

# Ensure all of the processes have finished
for j in jobs:
    j.join()
print("matchingTable shape " + str(np.array(matchingTable).shape))

# matchingTable = build_matching_table(tokenTable, tokenDictionary)
try:
    file = open(TMP_FOLDER + "matching_table",'wb')
    pickle.dump(tokenDictionary,file)
    print("'matching_table' created | Shape = " + str(np.array(matchingTable).shape))
    file.close()
except:
    print("failed to create file 'token_dictionary'")

# Extract processed datas spams and datas non spams
print(" Separating spams from non spams ...")
datasSpams = matchingTable[:spams_size]
datasNonSpams = matchingTable[spams_size:]

# Extract training datas
print(" Separating training data from testing data ...")
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
Y_datas = np.concatenate((Y_train, Y_test),axis=0)

np.savetxt(TMP_FOLDER + "datas_train.csv", np.array(datas_train), delimiter=";")
print("'datas_train.csv' created | Shape = " + str(np.array(datas_train).shape))

np.savetxt(TMP_FOLDER + "X_train.csv", np.array(X_train), delimiter=";")
print("'X_train.csv' created | Shape = " + str(np.array(X_train).shape))

np.savetxt(TMP_FOLDER + "Y_train.csv", np.array(Y_train), delimiter=";")
print("'Y_train.csv' created | Shape = " + str(np.array(Y_train).shape))


np.savetxt(TMP_FOLDER + "datas_test.csv", np.array(datas_test), delimiter=";")
print("'datas_test.csv' created | Shape = " + str(np.array(datas_test).shape))

np.savetxt( TMP_FOLDER + "X_test.csv", np.array(X_test), delimiter=";")
print("'X_test.csv' created | Shape = " + str(np.array(X_test).shape))

np.savetxt( TMP_FOLDER + "Y_test.csv", np.array(Y_test), delimiter=";")
print("'Y_test.csv' created | Shape = " + str(np.array(Y_test).shape))

np.savetxt( TMP_FOLDER + "token_dictionary.csv", np.array(tokenDictionary), delimiter=";",fmt="%s")
print("'token_dictionary.csv' created | Shape = " + str(np.array(tokenDictionary).shape))



######################################################
#######     CONCLUSION
######################################################