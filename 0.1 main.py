import numpy as np
import csv
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import matthews_corrcoef, confusion_matrix

PATH = "C:\\Users\\lprescott\\Documents\\Human proteome\\3CLpro GitHub\\"

# human proteome fasta
human_proteome_file_name = "1.3 UniProt_human_proteins_20350x_uniprot-reviewed_yes+AND+proteome_up000005640.fasta" # human proteome with 20,350 proteins

# CoV positive and negative cleavage samples
large_truePosFileName = "2.7 CoV_11x_cuts_some_repeats_4268x.txt"
#large_truePosFileName = "2.8 CoV_11x_cuts_no_repeats_762x.txt"
large_trueNegFileName_Qs = "2.9 CoV_true_negative_3CLpro_cuts_sites_Qs_no_repeats_or_XBJZs_17493x.txt"
large_trueNegFileName_Hs = "2.10 CoV_true_negative_3CLpro_cuts_sites_Hs_no_repeats_or_XBJZs_11421x.txt"

# pretrained NN to load
NN_to_load_file_name = "3.2 Final_NN_3538x_genomes_95_CV_0.98253_train_MCC_0.99759.sav"

# output file name
human_proteome_prediction_file_name = "4.1 NN_10N_in_1HL_applied_to_human_proteome.csv"



AAs = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V'] # 20AAs for one-hot encoding/decoding
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



def read_files(human_proteome_file_name, large_trueNegFileName_Qs, large_trueNegFileName_Hs, large_truePosFileName):
    # read human proteome fasta
    com = open(PATH + human_proteome_file_name, 'r')
    comIDList = []
    comSeqList = []
    tempSeq = ''
    for i, line in enumerate(com):
        if line.split(' ')[0][0] == '>':
            if tempSeq != '':
                comSeqList.append(tempSeq)
                tempSeq = ''
            comIDList.append(line.split(' ')[0].split('|')[1])
        else:
            tempSeq = tempSeq + line.replace("\n", "")
    comSeqList.append(tempSeq) #add one last seq
    
    # read new dataset known negatives file
    large_trueNegFile = open(PATH + large_trueNegFileName_Qs, 'r')
    large_trueNegList = []
    for i, line in enumerate(large_trueNegFile):
        large_trueNegList.append(line.replace("\n","").split("\t")[0])
    large_trueNegFile.close()
    large_trueNegFile_Hs = open(PATH + large_trueNegFileName_Hs, 'r')
    for i, line in enumerate(large_trueNegFile_Hs):
        large_trueNegList.append(line.replace("\n","").split("\t")[0])
    large_trueNegFile_Hs.close()
    
    # read new dataset known positives file
    large_truePosFile = open(PATH + large_truePosFileName, 'r')
    large_truePosList = []
    for i, line in enumerate(large_truePosFile):
        large_truePosList.append(line.replace("\n","").split("\t")[0])
    large_truePosFile.close()
        
    return comIDList, comSeqList, large_trueNegList, large_truePosList

# multiple oversampling methods ["None", "Naive", "SMOTE", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "ADASYN"]
def oversample(inputList, outputList, oversample_type):    
    if oversample_type == "None":
        return inputList, outputList
    elif oversample_type == "Naive":
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(random_state=np.random.randint(10000)).fit_resample(inputList, outputList)
    elif oversample_type == "SMOTE":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=np.random.randint(10000)).fit_resample(inputList, outputList)
    elif oversample_type == "BorderlineSMOTE":
        from imblearn.over_sampling import BorderlineSMOTE
        return BorderlineSMOTE(random_state=np.random.randint(10000)).fit_resample(inputList, outputList)
    elif oversample_type == "SVMSMOTE":
        from imblearn.over_sampling import SVMSMOTE
        return SVMSMOTE(random_state=np.random.randint(10000)).fit_resample(inputList, outputList)
    elif oversample_type.split("-")[0] == "KMeansSMOTE":
        from imblearn.over_sampling import KMeansSMOTE
        return KMeansSMOTE(k_neighbors=int(oversample_type.split("-")[1]), random_state=np.random.randint(10000)).fit_resample(inputList, outputList)
    elif oversample_type == "ADASYN":
        from imblearn.over_sampling import ADASYN
        return ADASYN(random_state=np.random.randint(10000)).fit_resample(inputList, outputList)
    else:
        print("Not a valid oversample type; try None, Naive, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, or ADASYN")

# encodes 2 (unbalanced) lists (pos and neg)
def encode_list_one_hot(trueNegList, truePosList):
    trueNegNumList = []
    for i, seq in enumerate(trueNegList):
        tempArray = np.zeros([10,20])
        for j, AA in enumerate(seq):
            for k, AA2 in enumerate(AAs):
                if AA == AA2:
                    tempArray[j][k] = 1
                    break
        trueNegNumList.append(tempArray.flatten())
    
    truePosNumList = []
    for i, seq in enumerate(truePosList):
        tempArray = np.zeros([10,20])
        for j, AA in enumerate(seq):
            for k, AA2 in enumerate(AAs):
                if AA == AA2:
                    tempArray[j][k] = 1
                    break
        truePosNumList.append(tempArray.flatten())
        
    unbalancedInputList = [*trueNegNumList, *truePosNumList]
    unbalancedOutputList = [*np.zeros(len(trueNegNumList)), *np.ones(len(truePosNumList))]
    print("Sample size =", len(unbalancedInputList))
    print("# positive samples =", int(sum(unbalancedOutputList)))
    return unbalancedInputList, unbalancedOutputList

def encode_list_physiochem(trueNegList, truePosList):
    trueNegNumList = []
    for seq in trueNegList:
        tempArray = []
        for AA in seq:
            tempArray.append(AA_vols[AAs.index(AA)])
            tempArray.append(AA_hydropathies_interface[AAs.index(AA)])
            tempArray.append(AA_hydropathies_octanol[AAs.index(AA)])
            tempArray.append(AA_pIs[AAs.index(AA)])
        trueNegNumList.append(tempArray)
    
    truePosNumList = []
    for seq in truePosList:
        tempArray = []
        for AA in seq:
            tempArray.append(AA_vols[AAs.index(AA)])
            tempArray.append(AA_hydropathies_interface[AAs.index(AA)])
            tempArray.append(AA_hydropathies_octanol[AAs.index(AA)])
            tempArray.append(AA_pIs[AAs.index(AA)])
        truePosNumList.append(tempArray)
    
    unbalancedInputList = [*trueNegNumList, *truePosNumList]
    unbalancedOutputList = [*np.zeros(len(trueNegNumList)), *np.ones(len(truePosNumList))]
    print("Sample size =", len(unbalancedInputList))
    print("# positive samples =", int(sum(unbalancedOutputList)))
    return unbalancedInputList, unbalancedOutputList

# similar to logistic regression encoding
def encode_list_pos_seq_logo(trueNegList, truePosList, pseudocount):
    tempSeqLogo1 = np.zeros((10, 20))
    for i, seq in enumerate(truePosList):
        for j, AA in enumerate(seq):
            if i != j:
                tempSeqLogo1[j][AAs.index(AA)] = tempSeqLogo1[j][AAs.index(AA)] + 1    
    tempSeqLogo2 = np.zeros((10,20))
    for i in range(10):
        for j in range(20):
            tempSeqLogo2[i][j] = (tempSeqLogo1[i][j] + pseudocount)/(sum(tempSeqLogo1[i]) + 20*pseudocount)
    
    truePosNumList = []
    for seq in truePosList:
        tempArray = []
        for i, AA in enumerate(seq):
            tempArray.append(tempSeqLogo2[i][AAs.index(AA)])
        truePosNumList.append(tempArray)
    trueNegNumList = []
    for seq in trueNegList:
        tempArray = []
        for i, AA in enumerate(seq):
            tempArray.append(tempSeqLogo2[i][AAs.index(AA)])
        trueNegNumList.append(tempArray)
    
    unbalancedInputList = [*trueNegNumList, *truePosNumList]
    unbalancedOutputList = [*np.zeros(len(trueNegNumList)), *np.ones(len(truePosNumList))]
    print("Sample size =", len(unbalancedInputList))
    print("# positive samples =", int(sum(unbalancedOutputList)))
    return unbalancedInputList, unbalancedOutputList

# similarlto naive Bayes classifier encoding
def encode_list_both_seq_logos(trueNegList, truePosList, pseudocount):
    tempPosSeqLogo1 = np.zeros((10, 20))
    for i, seq in enumerate(truePosList):
        for j, AA in enumerate(seq):
            if i != j:
                tempPosSeqLogo1[j][AAs.index(AA)] = tempPosSeqLogo1[j][AAs.index(AA)] + 1    
    tempPosSeqLogo2 = np.zeros((10,20))
    for i in range(10):
        for j in range(20):
            tempPosSeqLogo2[i][j] = (tempPosSeqLogo1[i][j] + pseudocount)/(sum(tempPosSeqLogo1[i]) + 20*pseudocount)
    tempNegSeqLogo1 = np.zeros((10, 20))
    for i, seq in enumerate(trueNegList):
        for j, AA in enumerate(seq):
            if i != j:
                tempNegSeqLogo1[j][AAs.index(AA)] = tempNegSeqLogo1[j][AAs.index(AA)] + 1    
    tempNegSeqLogo2 = np.zeros((10,20))
    for i in range(10):
        for j in range(20):
            tempNegSeqLogo2[i][j] = (tempNegSeqLogo1[i][j] + pseudocount)/(sum(tempNegSeqLogo1[i]) + 20*pseudocount)
    
    truePosNumList = []
    for seq in truePosList:
        tempArray = []
        for i, AA in enumerate(seq):
            tempArray.append(tempPosSeqLogo2[i][AAs.index(AA)])
            tempArray.append(tempNegSeqLogo2[i][AAs.index(AA)])
        truePosNumList.append(tempArray)
    trueNegNumList = []
    for seq in trueNegList:
        tempArray = []
        for i, AA in enumerate(seq):
            tempArray.append(tempPosSeqLogo2[i][AAs.index(AA)])
            tempArray.append(tempNegSeqLogo2[i][AAs.index(AA)])
        trueNegNumList.append(tempArray)
    
    unbalancedInputList = [*trueNegNumList, *truePosNumList]
    unbalancedOutputList = [*np.zeros(len(trueNegNumList)), *np.ones(len(truePosNumList))]
    print("Sample size =", len(unbalancedInputList))
    print("# positive samples =", int(sum(unbalancedOutputList)))
    return unbalancedInputList, unbalancedOutputList



# trains #-fold NNs with given hyperparameters and training data
def train_NN(train_inputList, train_outputList, test_input, test_output, HLs, a, oversample_type, num_cross_val, activation, solver):
    if num_cross_val < 0:
        print("#-fold CV must be >=1")
    elif num_cross_val == 0:
        train_input, train_output = oversample(train_inputList, train_outputList, oversample_type)
        
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        NN = []
        NN.append(MLPClassifier(solver=solver, activation=activation, alpha=a, hidden_layer_sizes=HLs, random_state=np.random.randint(10000), max_iter=10000, learning_rate='invscaling', verbose=True))
        NN[0].fit(train_input, train_output)
        tn, fp, fn, tp = confusion_matrix(test_output, NN[0].predict(test_input)).ravel()
        
        MCC = 0
        if tp+fp!=0 and tp+fn!=0 and tn+fp!=0 and tn+fn!=0:
            MCC = matthews_corrcoef(test_output, NN[0].predict(test_input))
        
        return NN, MCC
    else:
        cv = ShuffleSplit(n_splits=num_cross_val, test_size=1.0/num_cross_val, random_state=np.random.randint(10000))
        
        train_input = []
        train_output = []
        test_input = []
        test_output = []
        for train_index, test_index in cv.split(train_inputList):
            temp_train_input = []
            temp_train_output = []
            temp_test_input = []
            temp_test_output = []
            
            for index in train_index:
                temp_train_input.append(train_inputList[index])
                temp_train_output.append(train_outputList[index])
            for index in test_index:
                temp_test_input.append(train_inputList[index])
                temp_test_output.append(train_outputList[index])
            
            train_input.append(temp_train_input)
            train_output.append(temp_train_output)
            test_input.append(temp_test_input)
            test_output.append(temp_test_output)
        
        for i in range(num_cross_val):
            train_input[i], train_output[i] = oversample(train_input[i], train_output[i], oversample_type)
        
        NN = []
        
        tn = np.zeros(num_cross_val)
        fp = np.zeros(num_cross_val)
        fn = np.zeros(num_cross_val)
        tp = np.zeros(num_cross_val)
        for i in range(num_cross_val):
            NN.append(MLPClassifier(solver=solver, activation=activation, alpha=a, hidden_layer_sizes=HLs, random_state=np.random.randint(10000), max_iter=10000, learning_rate='invscaling', verbose=False))
            NN[i].fit(train_input[i], train_output[i])
            tn[i], fp[i], fn[i], tp[i] = confusion_matrix(test_output[i], NN[i].predict(test_input[i])).ravel()
        
        MCC = np.zeros(num_cross_val)
        for i in range(num_cross_val):
            if tp[i]+fp[i]!=0 and tp[i]+fn[i]!=0 and tn[i]+fp[i]!=0 and tn[i]+fn[i]!=0:
                MCC[i] = matthews_corrcoef(test_output[i], NN[i].predict(test_input[i]))
                
        return NN, MCC.mean()

# apply NN to test data and compare with given outputs
def test_NN(NN, inputList, outputList):
    final_output = []    
    for i, full_test_input in enumerate(inputList):
        score = []
        for j, single_NN in enumerate(NN):
            score.append(round(float(NN[j].predict_proba([full_test_input]).flatten()[1]),10))
        final_output.append(round(sum(score)/len(score),0))
    final_MCC = matthews_corrcoef(outputList, final_output)
    return final_MCC

# apply NN to human proteome
def test_human_proteome(NN, comIDList, comSeqList, human_proteome_file_name):
    with open(PATH + human_proteome_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["UniProt ID","Position (AAs)","Sequence","NN score"])
        
        for i, seq in enumerate(tqdm(comSeqList)):
            for j, AA in enumerate(seq):
                if AA=='Q' and j>=4 and j<len(seq)-6:
                    seqToNumArray = encode(seq[j-4:j+6])
                                        
                    score = []
                    for k, single_NN in enumerate(NN):
                        score.append(round(NN[k].predict_proba([seqToNumArray]).flatten()[1], 0))
                    score = sum(score)/len(score)
                    
                    if score >= 0.5:
                        writer.writerow([comIDList[i], j+1, seq[j-4:j+6], score])

# generate a single protein with 1000 potential cut sites centered at Qs
def generate_random_protein():
    temp_AAs = 'LSEAGPVKRTQDIFNYHCMW'
    # AA frequencies in humans
    probs = [0.1,0.0833,0.0709,0.0701,0.0658,0.0631,0.0596,0.0572,0.0564,0.0536,0.0477,0.0473,0.0433,0.0365,0.0358,0.0266,0.0263,0.0230,0.0213,0.0122]
    # L is actually 0.0997, but they didn't add up to 1
    
    randomProtein = ''
    randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
    randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
    randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
    randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
    randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
    
    for numNonrandomQs in range(1000):
        randomProtein = randomProtein + 'Q' # e.g. 5, 11, 17, etc.
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
    
    return randomProtein

# apply NN to 100x proteins each with 1000 potential cut sites centered at Qs. Print number that are and are not cut
def test_random_proteins(NN):
    cuts_count = 0
    not_cuts_count = 0
    for i in tqdm(range(100)):
        temp_random_protein = generate_random_protein()
        for j in range(1000):
            temp_vector = encode(temp_random_protein[6*j+1 : 6*j+11])
            
            score = []
            for i, single_NN in enumerate(NN):
                score.append(round(NN[i].predict_proba([temp_vector]).flatten()[1], 0))
            avg_score = round(sum(score)/len(score),0)
            
            if avg_score == 1:
                cuts_count = cuts_count + 1
            elif avg_score == 0:
                not_cuts_count = not_cuts_count + 1
    print("Random frequency =", cuts_count/(cuts_count + not_cuts_count))

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

# Loop over hyperparameters [test/train split fraction, solver, activation function, #-fold CV, oversampling method, neurons in hidden layers, regularization]
# For ever condition, repeat 25 train/test cycles to get error bars
def hyperparameter_optimization(large_unevenInputList, large_unevenOutputList):
    for i in range(25):
        for split in [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
            train_input, test_input, train_output, test_output = train_test_split(large_unevenInputList, large_unevenOutputList, test_size=split, random_state=np.random.randint(10000))
            for solver in ['lbfgs']:#, 'sgd', 'adam']:
                for activation in ['tanh']:#'identity', 'logistic', 'tanh', 'relu']:
                    for num_CV in [3]:#[2,3,4,5]:
                        for os in ["None"]:#["None","Naive","SMOTE","ADASYN","BorderlineSMOTE","SVMSMOTE","KMeansSMOTE"]:
                           for HLs in [10]:#10, (10,2)]:#, (2,2), (3,3), (4,4), (5,5), (10,10)]:
                                for a in [0.00001]:
                                    NN, CV_MCC = train_NN(train_input, train_output, test_input, test_output, HLs, a, os, num_CV, activation, solver)
                                    final_MCC_train = test_NN(NN, train_input, train_output)
                                    final_MCC_test = test_NN(NN, test_input, test_output)
                                    print(i, split, len(train_input), len(test_input), solver, activation, num_CV, os, HLs, a, "CV, 80% final, 20% final MCCs =", round(CV_MCC,5), round(final_MCC_train,5), round(final_MCC_test,5)) # note CV is on potentialy oversampled data

# train many NNs with the optimum hyperparameters to maximize avg CV MCC. Pickle best networks for later
def train_many_NNs(large_unevenInputList, large_unevenOutputList, min_MCC):
    temp_train_MCC = 0
    count = 1
    while temp_train_MCC < min_MCC:
        print(count)
        train_input, test_input, train_output, test_output = train_test_split(large_unevenInputList, large_unevenOutputList, test_size=0.9, random_state=np.random.randint(10000))
        print(len(train_input), len(test_input))
        NN, train_MCC = train_NN(train_input, train_output, test_input, test_output, 10, 0.00001, 'Naive', 3, 'tanh', 'lbfgs')
        if train_MCC > temp_train_MCC:
            temp_train_MCC = train_MCC
            print("train_MCC =", round(train_MCC,5))
            
            test_MCC = test_NN(NN, test_input, test_output)
            print("test_MCC =", round(test_MCC,5))
            pickle.dump(NN, open(PATH + "3.3 NN_3538x_genomes_" + str(count) + "_train_MCC_" + str(round(train_MCC,5)) + "_test_MCC_" + str(round(test_MCC,5)) + ".sav", 'wb'))
        count = count + 1

def main():
    # read files [human_proteome, CoV 4268x positives (mainly Qs), CoV negative 17493x Qs and 11421x Hs]
    comIDList, comSeqList, large_trueNegList, large_truePosList = read_files(human_proteome_file_name, large_trueNegFileName_Qs, large_trueNegFileName_Hs, large_truePosFileName)
    print("Human proteome size =", len(comIDList))
    
    # encode CoV inuts as one-hot, 4x physiochem parameters, or positive and/or negative sequence logo probabilities (with small but nonzero pseudocount = 0.01)
    large_unevenInputList, large_unevenOutputList = encode_list_one_hot(large_trueNegList, large_truePosList)
    #large_unevenInputList, large_unevenOutputList = encode_list_physiochem(large_trueNegList, large_truePosList)
    #large_unevenInputList, large_unevenOutputList = encode_list_pos_seq_logo(large_trueNegList, large_truePosList, 0.01)
    #large_unevenInputList, large_unevenOutputList = encode_list_both_seq_logos(large_trueNegList, large_truePosList, 0.01)
    
    # hyperparameter optimization many nested for loops
    #hyperparameter_optimization(large_unevenInputList, large_unevenOutputList)
    
    # train many CV'd NNs (with optimized hyperparameters) until minimum MCC is achieved (currently impossible 1.01)
    #train_many_NNs(large_unevenInputList, large_unevenOutputList, 1.01)
    
    # or load saved NN to avoid retraining cycles
    loaded_NN = pickle.load(open(PATH + NN_to_load_file_name, "rb"))
    final_MCC = test_NN(loaded_NN, large_unevenInputList, large_unevenOutputList)
    print("Final MCC", final_MCC)
    
    # test trained or loaded NN(s) on 20350 human proteins
    print("Testing NN(s) on human proteome")
    test_human_proteome(loaded_NN, comIDList, comSeqList, human_proteome_prediction_file_name)
    
    # test trained or loaded NN(s) on 100,000x Qs to estimate random cleavage rate
    print("testing NN(s) on random proteins")
    test_random_proteins(loaded_NN)

main()







