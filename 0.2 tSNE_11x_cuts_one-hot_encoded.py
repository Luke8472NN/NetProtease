import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, ShuffleSplit
from sklearn.metrics import matthews_corrcoef, f1_score, make_scorer, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import random

PATH = "C:\\Users\\lprescott\\Documents\\Human proteome\\3CLpro GitHub\\"

# CoV positive cleavage sequences and numbers file
#fileName = "2.7 CoV_11x_cuts_some_repeats_4268x.txt"
fileName = "2.8 CoV_11x_cuts_no_repeats_762x.txt"

AAs = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

# read CoV cleavage sequences and numbers file
def read_files():
    file = open(PATH + fileName, 'r')
    labelsList = []
    dataList = []
    for i, line in enumerate(file):
        dataList.append(line.replace("\n","").split("\t")[0])
        labelsList.append(line.replace("\n","").split("\t")[1])
    file.close()
    return labelsList, dataList

# encode a single 10char string into a 200bit array
def encode(string_10char):
    tempArray = np.zeros([10,20])
    for i, AA in enumerate(string_10char):
        for j, AA2 in enumerate(AAs):
            if AA == AA2:
                tempArray[i][j] = 1
                break
    return tempArray.flatten()

# decode a single 200bit array into a 10char string
def decode(array_200bit):
    temp_string = ""
    for j, bit in enumerate(array_200bit): # 0 or 1 x200
        if bit == 1:
            temp_string = temp_string + AAs[j%20]
    return temp_string

# convert arbitrary dimensional vectors to 2D for plotting
def tSNE(labels, vectors):
    labels = np.asarray(labels)
    vectors = np.asarray(vectors)
    
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=2, random_state=0) # reduce to 2D
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

# make scatterplot of 2D tSNE outputs
def plot_with_matplotlib(x_vals, y_vals, labels):    
    random.seed(0)
    
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    indices = list(range(len(labels)))

    selected_indices = random.sample(indices, 100) # only label 100 points
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]), fontsize=15)

def main():
    labelsList, dataList = read_files()
    print("Read file")
    
    x_vals, y_vals, labels = tSNE(labelsList, [encode(seq) for seq in dataList])
    print("tSNE done")
    
    plot_with_matplotlib(x_vals, y_vals, labels)
        
    # write file for external analysis
    #with open(PATH + 'tSNE_CoV_11x_cuts_no_repeats_762x.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    for i in range(len(x_vals)):
    #        writer.writerow([x_vals[i], y_vals[i], labels[i]])




main()







