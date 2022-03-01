import numpy as np
import random
import matplotlib.pyplot as plt
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceMatrix, DistanceTreeConstructor
from tqdm import tqdm
import csv

from Bio.Blast import NCBIWWW
from Bio import SeqIO
from Bio.Blast import NCBIXML
from Bio import pairwise2
from Bio import Align
from Bio import AlignIO
import numpy as np
from Bio.pairwise2 import format_alignment
from Bio.SubsMat import MatrixInfo as matlist
import math

PATH = "C:\\Users\\TEMP\\Documents\\TEMP_13Aug21\\Human proteome\\Togaviridae\\"
#file_name = "301x_PLpros_and_cuts.txt"
file_name_2 = "2.14 nsP2pro_alignment_scores.csv"

# openning 10, extension 0.2

# read individual and subgenera-combined blosum62 similarity matrices between 3CLpros and between cleavages
def read_file_1(file_name):
    file = open(PATH + file_name, 'r')
    IDs = []
    PLpros = []
    CS1s = []
    CS2s = []
    CS3s = []
    for i, line in enumerate(file):
        if i>0:
            IDs.append(line.split('\t')[0])
            PLpros.append(line.split('\t')[1])
            CS1s.append(line.split('\t')[2])
            CS2s.append(line.split('\t')[3])
            CS3s.append(line.split('\t')[4].replace('\n',''))
    file.close()
    return IDs, PLpros, CS1s, CS2s, CS3s

def read_file_2(file_name):
    file = open(PATH + file_name, 'r')
    PLpro_scores = []
    for i, line in enumerate(file):
        if i == 1:
            labels = line.replace("\n","").split(",")[2:43]
        elif i>=2:
            PLpro_scores.append([float(temp) for temp in line.replace("\n","").split(",")[2:43]])
    file.close()
    
    return labels, PLpro_scores
    

def main():
    IDs, PLpros, CS1s, CS2s, CS3s = read_file_1(file_name)
    
    # align CS1s
    CS1_alignment_matrix = np.array(len(CS1s)*[len(CS1s)*[0.0]])    
    for i in tqdm(range(0, len(CS1s))):
        for j in range(0, len(CS1s)):
            if i<=j:
                CS1_alignment_matrix[i,j] = pairwise2.align.globalds(CS1s[i], CS1s[j], matlist.blosum62, -1000000, 0, score_only=True)
    
    # align CS2s
    CS2_alignment_matrix = np.array(len(CS2s)*[len(CS2s)*[0.0]])    
    for i in tqdm(range(0, len(CS2s))):
        for j in range(0, len(CS2s)):
            if i<=j:
                CS2_alignment_matrix[i,j] = pairwise2.align.globalds(CS2s[i], CS2s[j], matlist.blosum62, -1000000, 0, score_only=True)
    
    # align CS3s
    CS3_alignment_matrix = np.array(len(CS3s)*[len(CS3s)*[0.0]])    
    for i in tqdm(range(0, len(CS3s))):
        for j in range(0, len(CS3s)):
            if i<=j:
                CS3_alignment_matrix[i,j] = pairwise2.align.globalds(CS3s[i], CS3s[j], matlist.blosum62, -1000000, 0, score_only=True)
    
    print(CS1_alignment_matrix)
    print(CS2_alignment_matrix)
    print(CS3_alignment_matrix)
    
    with open(PATH + 'PLpro_cut_alignments_1_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for i in range(len(CS1_alignment_matrix)):
            writer.writerow(CS1_alignment_matrix[i])
        writer.writerow(["space"])
        for i in range(len(CS2_alignment_matrix)):
            writer.writerow(CS2_alignment_matrix[i])
        writer.writerow(["space"])
        for i in range(len(CS3_alignment_matrix)):
            writer.writerow(CS3_alignment_matrix[i])
        writer.writerow(["space"])
        
    
    # align PLpros
    PLpro_alignment_matrix = np.array(len(PLpros)*[len(PLpros)*[0.0]])    
    for i in tqdm(range(0, len(PLpros))):
        for j in tqdm(range(0, len(PLpros))):
            if i<=j:
                PLpro_alignment_matrix[i,j] = pairwise2.align.globalds(PLpros[i], PLpros[j], matlist.blosum62, -10, -0.2, score_only=True)
    
    print(PLpro_alignment_matrix)
    
    with open(PATH + 'PLpro_seq_alignments_1_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for i in range(len(PLpro_alignment_matrix)):
            writer.writerow(PLpro_alignment_matrix[i])



def main2():
    labels, PLpro_scores = read_file_2(file_name_2)
    
    # convert blosum62-gaps similarities to unscaled distances
    for i in range(len(PLpro_scores)):
        for j in range(len(PLpro_scores[i])):
            PLpro_scores[i][j] = 11000 - PLpro_scores[i][j]
    
    # convert to lower triangle matrices
    PLpro_scores_lower = []
    for i in range(len(PLpro_scores)):
        PLpro_scores_lower.append(PLpro_scores[i][0:i+1])
        
    # create DistanceMatrix objects
    dm_PLpro = DistanceMatrix(names=labels, matrix=PLpro_scores_lower)
        
    # create and ladderize trees
    tree_PLpro = DistanceTreeConstructor().nj(dm_PLpro)
    tree_PLpro.ladderize()
        
    # remove inner branch labels
    for i, clade in enumerate(tree_PLpro.find_clades()):
        if clade.name[0:5] == "Inner":
            clade.name = " "*i
        clade.branch_length = 1
    
    # display trees
    print("PLpro phylogeny")
    fig = plt.figure(figsize=(8, 5), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree_PLpro, do_show=False, axes=axes)

#main()
main2()





