import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, ShuffleSplit
from sklearn.metrics import matthews_corrcoef, f1_score, make_scorer, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

PATH = "C:\\Users\\TEMP\\Documents\\TEMP_13Aug21\\Human proteome\\Togaviridae\\"

# CoV positive cleavage sequences and numbers file
fileName = "Togaviridae_positives_no_repeats_93x.txt"

AAs = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
AA_vols = [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7, 166.7, 168.6, 162.9, 189.9, 112.7, 89.0, 116.1, 227.8, 193.6, 140.0] # in cubed Angstroms
AA_hydropathies_interface = [0.17, 0.81, 0.42, 1.23, -0.24, 0.58, 2.02, 0.01, 0.17, -0.31, -0.56, 0.99, -0.23, -1.13, 0.45, 0.13, 0.14, -1.85, -0.94, 0.07]
AA_hydropathies_octanol =   [0.50, 1.81, 0.85, 3.64, -0.02, 0.77, 3.63, 1.15, 0.11, -1.12, -1.25, 2.80, -0.67, -1.71, 0.14, 0.46, 0.25, -2.09, -0.71, -0.46]
# Note aspartic acid, glutamic acid are negative, histidine is neutral, and lysine and arginine are positive at pH 7. Hydropathy values differ between charged species
AA_pIs = [6.00, 10.76, 5.41, 2.77, 5.07, 3.22, 5.65, 5.97, 7.59, 6.02, 5.98, 9.74, 5.74, 5.48, 6.30, 5.68, 5.60, 5.89, 5.66, 5.96]

#normalize these 4x cales
AA_vols                   = [(temp-min(AA_vols                  ))/(max(AA_vols                  )-min(AA_vols                  )) for temp in AA_vols]
AA_hydropathies_interface = [(temp-min(AA_hydropathies_interface))/(max(AA_hydropathies_interface)-min(AA_hydropathies_interface)) for temp in AA_hydropathies_interface]
AA_hydropathies_octanol   = [(temp-min(AA_hydropathies_octanol  ))/(max(AA_hydropathies_octanol  )-min(AA_hydropathies_octanol  )) for temp in AA_hydropathies_octanol]
AA_pIs                    = [(temp-min(AA_pIs                   ))/(max(AA_pIs                   )-min(AA_pIs                   )) for temp in AA_pIs]


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
def encode_one_hot(string_10char):
    tempArray = np.zeros([10,20])
    for i, AA in enumerate(string_10char):
        for j, AA2 in enumerate(AAs):
            if AA == AA2:
                tempArray[i][j] = 1
                break
    return tempArray.flatten()

def encode_physiochem(string_10char):
    tempArray = []
    for AA in string_10char:
        tempArray.append(AA_vols[AAs.index(AA)])
        tempArray.append(AA_hydropathies_interface[AAs.index(AA)])
        tempArray.append(AA_hydropathies_octanol[AAs.index(AA)])
        tempArray.append(AA_pIs[AAs.index(AA)])
    return tempArray

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
    
    x_vals, y_vals, labels = tSNE(labelsList, [encode_physiochem(seq) for seq in dataList])
    print("tSNE done")
    
    #plot_with_matplotlib(x_vals, y_vals, labels)
    
    # write file for external analysis
    with open(PATH + 'tSNE_CoV_3x_cuts_no_repeats_93x.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(x_vals)):
            writer.writerow([x_vals[i], y_vals[i], labels[i], dataList[i]])




main()







