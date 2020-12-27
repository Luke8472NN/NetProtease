import numpy as np
import random
import matplotlib.pyplot as plt
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceMatrix, DistanceTreeConstructor

PATH = "C:\\Users\\lprescott\\Documents\\Human proteome\\3CLpro GitHub\\"
file2 = "2.12 3CLpro_cut_blosum62_individual_scores_379x379.csv" # all 379x orf1abs
file1 = "2.13 3CLpro_pro_blosum62_individual_scores_379x379.csv" # all 379x orf1abs
file3 = "2.14 3CLpro_pro_and_cut_blosum62_combo_scores_2x19x19.csv" # combined as avg subgenera scores

# read individual and subgenera-combined blosum62 similarity matrices between 3CLpros and between cleavages
def read_files():
    f1 = open(PATH + file1, 'r')
    list1labels = []
    list1scores = []
    for i, line in enumerate(f1):
        if i>0:
            list1labels.append(line.replace("\n","").split(",")[0])
            list1scores.append(line.replace("\n","").split(",")[1:390])
    f1.close()
    
    f2 = open(PATH + file2, 'r')
    list2labels = []
    list2scores = []
    for i, line in enumerate(f2):
        if i>0:
            list2labels.append(line.replace("\n","").split(",")[0][:-1])
            list2scores.append(line.replace("\n","").split(",")[1:390])
    f2.close()
    
    f3 = open(PATH + file3, 'r')
    list3labels = []
    list3scores = []
    list4labels = []
    list4scores = []
    for i, line in enumerate(f3):
        if i>=2 and i<=20:
            list3labels.append(line.replace("\n","").split(",")[1])
            list3scores.append([int(temp) for temp in line.replace("\n","").split(",")[2:21]])
        elif i>=24:
            list4labels.append(line.replace("\n","").split(",")[1])
            list4scores.append([int(temp) for temp in line.replace("\n","").split(",")[2:21]])
    f3.close()
    
    return list1labels, list1scores, list2labels, list2scores, list3labels, list3scores, list4labels, list4scores

def main():
    list1labels, list1scores, list2labels, list2scores, list3labels, list3scores, list4labels, list4scores = read_files()
    
    for i in range(len(list3scores)):
        for j in range(len(list3scores[i])):
            list3scores[i][j] = 1/list3scores[i][j]
    for i in range(len(list4scores)):
        for j in range(len(list4scores[i])):
            list4scores[i][j] = 1/list4scores[i][j]
        
    list3scoreslower = []
    for i in range(len(list3scores)):
        list3scoreslower.append(list3scores[i][0:i+1])
    list4scoreslower = []
    for i in range(len(list4scores)):
        list4scoreslower.append(list4scores[i][0:i+1])
    
    dm_pro = DistanceMatrix(names=list3labels, matrix=list3scoreslower)
    dm_cut = DistanceMatrix(names=list4labels, matrix=list4scoreslower)
    
    tree_pro = DistanceTreeConstructor().nj(dm_pro)
    tree_cut = DistanceTreeConstructor().nj(dm_cut)
    tree_pro.ladderize()
    tree_cut.ladderize()
    
    for i, clade in enumerate(tree_pro.find_clades()):
        if clade.name[0:5] == "Inner":
            clade.name = " "*i
        clade.branch_length = 1
    for i, clade in enumerate(tree_cut.find_clades()):
        if clade.name[0:5] == "Inner":
            clade.name = " "*i
        clade.branch_length = 1
    
    print("Protease phylogeny")
    fig = plt.figure(figsize=(8, 5), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree_pro, do_show=False, axes=axes)
    
    print("Cleavage phylogeny")
    fig = plt.figure(figsize=(8, 5), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree_cut, do_show=False, axes=axes)

main()









