import numpy as np
import csv
import matplotlib.pyplot as plt
import math

PATH = "C:\\Users\\lprescott\\Documents\\Human proteome\\3CLpro GitHub\\"
file_name = "2.7 CoV_11x_cuts_some_repeats_4268x.txt"


# plot 10x11 matrix of information content vs 11x cleavages and position (10AAs)
temp_matrix = [[2.18907574,3.35029368,2.1541747,4.16549526,4.1871325,3.31971483,4.29599255,2.92076438,2.65797509,3.9412865],[4.04671644,3.82206965,1.89955506,3.35727698,4.321928,3.5845021,2.24197302,1.7831735,2.45347937,2.26832623],[3.32455106,3.24753961,3.11935915,3.4625011,4.321928,4.15087426,3.92072727,3.70593568,3.11705803,3.89265773],[1.89824027,2.8613175,2.85552549,4.22234665,4.20653761,3.41952053,3.04907984,2.55386149,2.79822237,2.85662499],[1.62730401,2.62270501,2.54109861,4.10023897,4.24931861,4.321928,4.29599255,4.29599255,3.2669216,2.36799535],[3.207049,3.7986995,3.06913355,4.24931861,4.1355035,3.25613306,3.15386694,2.01837089,1.65667493,3.13808251],[1.71124429,2.24090144,1.66885497,2.24937786,4.20653761,3.18861531,1.43755694,2.49446887,1.26927574,1.54660387],[2.94889627,3.2327976,2.91633506,3.90144204,4.321928,3.36218854,2.53497463,4.27523472,2.03341949,4.29599255],[1.43326973,1.83952346,1.30407828,3.93966517,4.02843867,2.34068117,1.8524997,1.56863632,1.82733782,2.03938335],[2.57547658,2.16748641,2.27171712,3.792357,4.321928,3.27260483,3.32518387,3.32375665,4.11379016,3.08989821],[4.321928,4.27006676,2.99711809,4.15803803,4.321928,3.34231919,2.68604611,2.00702936,1.8159355,2.01453375]]

fig, ax = plt.subplots(figsize=(10,11), dpi=50)
im = ax.imshow(temp_matrix, cmap="viridis")
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(11))
ax.set_xticklabels(range(1,11), fontsize=20)
ax.set_yticklabels(range(1,12), fontsize=20)
ax.set_ylim(len(temp_matrix)-0.5, -0.5)

for i in range(11):
    for j in range(10):
        text = ax.text(j, i, round(np.array(temp_matrix)[i, j],1), fontsize="20", ha="center", va="center", color="w")

ax.set_title("Information Content vs Cleavage and Position", fontsize=23)
ax.set_xlabel("Position in Cleavage", fontsize=20)
ax.set_ylabel("Number Cleavage in Polyprotein", fontsize=20)
plt.show()







AA_string = "ARNDCEQGHILKMFPSTWYV"
pseudocount = 0.01 # small but nonzero is optimal

# read file with 4268x CoV positives
seqList = []
numList = []
com = open(PATH + file_name, 'r')
for i, line in enumerate(com):
    seqList.append(line.split("\t")[0].replace("\n",""))
    numList.append(line.split("\t")[1].replace("\n",""))
com.close()
print("# seqs =", len(seqList))

# make 10x20 PFM
tempSeqLogo1 = np.zeros((10, 20))
for i, seq in enumerate(seqList):
    for j, AA in enumerate(seq):
        if i != j:
            tempSeqLogo1[j][AA_string.index(AA)] = tempSeqLogo1[j][AA_string.index(AA)] + 1

# normalize the PFM to a PWM with pseudocount
tempSeqLogo2 = np.zeros((10,20))
for i in range(10):
    for j in range(20):
        tempSeqLogo2[i][j] = (tempSeqLogo1[i][j] + pseudocount)/(sum(tempSeqLogo1[i]) + 20*pseudocount)

# combine this sequence logo's 20D columns into 1 number representing information content i.e. total height of letters when displayed
tempSeqLogo3 = np.zeros(10)
for i in range(10):
    tempSum = 0
    for j in range(20):
        if tempSeqLogo2[i][j] > 0:
            tempSum = tempSum - tempSeqLogo2[i][j]*np.log2(tempSeqLogo2[i][j])
    tempSeqLogo3[i] = np.log2(20) - (tempSum + ((20 - 1)/(2*len(seqList)*np.log(2)))) #tempSum
print("1D Hs =", tempSeqLogo3)



# 2D PFM in 10x10x400 matrix
temp2DSeqLogo1 = np.zeros((10,10,400))
for i, seq in enumerate(seqList):
    for j in range(10):
        for k in range(10):
            temp2DSeqLogo1[j][k][20*AA_string.index(seq[j]) + AA_string.index(seq[k])] = temp2DSeqLogo1[j][k][20*AA_string.index(seq[j]) + AA_string.index(seq[k])] + 1

# normalize PFM to PWM with same pseudocount
temp2DSeqLogo2 = np.zeros((10,10,400))
for i in range(10):
    for j in range(10):
        for k in range(400):
            temp2DSeqLogo2[i][j][k] = (temp2DSeqLogo1[i][j][k] + pseudocount)/(sum(temp2DSeqLogo1[i][j]) + 20*pseudocount)

# combine 1D and 2D PWMs to different masures of mutual info
# JE = joint entropy, (N)MI = (normalized) mutual info, ECC = entropy correlation coefficient
temp2DSeqLogo3_JE = np.zeros((10,10))
temp2DSeqLogo3_MI = np.zeros((10,10))
temp2DSeqLogo3_NMI = np.zeros((10,10))
temp2DSeqLogo3_ECC = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        if i == j:
            temp2DSeqLogo3_JE[i][j] = math.nan
            temp2DSeqLogo3_MI[i][j] = math.nan
            temp2DSeqLogo3_NMI[i][j] = math.nan
            temp2DSeqLogo3_ECC[i][j] = math.nan
        else:
            for k in range(400):
                temp2DSeqLogo3_JE[i][j] = temp2DSeqLogo3_JE[i][j] - temp2DSeqLogo2[i][j][k]*np.log2(temp2DSeqLogo2[i][j][k])
                temp2DSeqLogo3_MI[i][j] = temp2DSeqLogo3_MI[i][j] + temp2DSeqLogo2[i][j][k]*np.log2(temp2DSeqLogo2[i][j][k]/(tempSeqLogo2[i][int(k/20)]*tempSeqLogo2[j][k%20]))
            temp2DSeqLogo3_NMI[i][j] = 1 + (temp2DSeqLogo3_MI[i][j]/temp2DSeqLogo3_JE[i][j])
            temp2DSeqLogo3_ECC[i][j] = 2*(1 - 1/temp2DSeqLogo3_NMI[i][j])



# plot 10x10 ECC matrix
fig, ax = plt.subplots(figsize=(10,10), dpi=50)
im = ax.imshow(temp2DSeqLogo3_ECC, cmap="viridis")
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(range(1,11), fontsize=20)
ax.set_yticklabels(range(1,11), fontsize=20)
ax.set_ylim(len(temp2DSeqLogo3_ECC)-0.5, -0.5)

for i in range(10):
    for j in range(10):
        text = ax.text(j, i, round(temp2DSeqLogo3_ECC[i, j],1), fontsize="20", ha="center", va="center", color="w")

ax.set_title("2D Sequence Logo", fontsize=30)
ax.set_xlabel("Position in Cleavage", fontsize=20)
ax.set_ylabel("Position in Cleavage", fontsize=20)
plt.show()


