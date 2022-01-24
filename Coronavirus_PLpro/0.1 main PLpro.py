import numpy as np
import csv
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

PATH = "C:\\Users\\lprescott\\Documents\\Human proteome\\PLpro\\"

# human proteome fasta
human_proteome_file_name = "1.2 UniProt_human_proteins_20350x_uniprot-reviewed_yes+AND+proteome_up000005640.fasta" # human proteome with 20,350 proteins

# CoV positive and negative cleavage samples
large_truePosFileName = "2.3 PLpro_cuts_CoV_list_no_repeats_463x.txt" # 468x seqs
large_trueNegFileName = "2.4 PLpro_negatives_2ormore_Gs_and_As_no_repeats_33185x.txt" # 35,345x seqs

# output file name
human_proteome_prediction_file_name = "4.0 NN_10N_in_1HL_applied_to_human_proteome_10CV.csv"



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



def read_files(human_proteome_file_name, large_trueNegFileName, large_truePosFileName):
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
    large_trueNegFile = open(PATH + large_trueNegFileName, 'r')
    large_trueNegList = []
    large_trueNegSubgenusList = []
    for i, line in enumerate(large_trueNegFile):
        large_trueNegList.append(line.split("\t")[0])
        large_trueNegSubgenusList.append(line.split("\t")[1].replace("\n",""))
    large_trueNegFile.close()
    
    # read new dataset known positives file
    large_truePosFile = open(PATH + large_truePosFileName, 'r')
    large_truePosList = []
    large_truePosSubgenusList = []
    for i, line in enumerate(large_truePosFile):
        large_truePosList.append(line.split("\t")[0])
        large_truePosSubgenusList.append(line.split("\t")[2].replace("\n",""))
    large_truePosFile.close()
    
    return comIDList, comSeqList, large_trueNegList, large_trueNegSubgenusList, large_truePosList, large_truePosSubgenusList

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
        
    unbalancedInputList = [*truePosNumList, *trueNegNumList]
    unbalancedOutputList = [*np.ones(len(truePosNumList)), *np.zeros(len(trueNegNumList))]
    #print("Sample size =", len(unbalancedInputList))
    #print("# positive samples =", int(sum(unbalancedOutputList)))
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
def train_NN(train_inputList, train_outputList, HLs, a, oversample_type, num_cross_val, activation, solver):
    if num_cross_val < 0:
        print("#-fold CV must be >=1")
    elif num_cross_val == 0:
        train_input, train_output = oversample(train_inputList, train_outputList, oversample_type)
        
        #tn, fp, fn, tp = 0, 0, 0, 0
        NN = []
        NN.append(MLPClassifier(solver=solver, activation=activation, alpha=a, hidden_layer_sizes=HLs, random_state=np.random.randint(10000), max_iter=10000, learning_rate='invscaling', verbose=False))
        NN[0].fit(train_input, train_output)
        #tn, fp, fn, tp = confusion_matrix(test_output, NN[0].predict(test_input)).ravel()
        
        MCC = 0
        #if tp+fp!=0 and tp+fn!=0 and tn+fp!=0 and tn+fn!=0:
        #    MCC = matthews_corrcoef(test_output, NN[0].predict(test_input))
        
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
        for i in tqdm(range(num_cross_val)):
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
    for i, full_test_input in enumerate(tqdm(inputList)):
        score = []
        for j, single_NN in enumerate(NN):
            score.append(round(float(NN[j].predict_proba([full_test_input]).flatten()[1]),10))
        final_output.append(round(sum(score)/len(score),0))
        
        #if round(sum(score)/len(score),0) != outputList[i]:
        #    print(decode(inputList[i]), sum(score)/len(score), outputList[i])
    
    final_MCC = matthews_corrcoef(outputList, final_output)
    return final_MCC

# apply NN to human proteome
def test_human_proteome(NN, comIDList, comSeqList, human_proteome_prediction_file_name):
    with open(PATH + human_proteome_prediction_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["UniProt ID","Position (AAs)","Sequence","NN score"])
        
        for i, seq in enumerate(tqdm(comSeqList)):
            if len(seq)>=10:
                for j, AA in enumerate(seq):
                    if j<len(seq)-10 and seq[j+3:j+6].count("G") + seq[j+3:j+6].count("A") >=2:
                        score = []
                        for k, single_NN in enumerate(NN):
                            score.append(round(NN[k].predict_proba([encode(seq[j:j+10])]).flatten()[1], 0))
                        score = sum(score)/len(score)
                        
                        if score >= 0.5:
                            writer.writerow([comIDList[i], j+1, seq[j:j+10], score])

# generate a single protein with 1000 potential cut sites centered at Qs
def generate_random_protein():
    temp_AAs = 'LSEAGPVKRTQDIFNYHCMW'
    # AA frequencies in humans
    probs = [0.1,0.0833,0.0709,0.0701,0.0658,0.0631,0.0596,0.0572,0.0564,0.0536,0.0477,0.0473,0.0433,0.0365,0.0358,0.0266,0.0263,0.0230,0.0213,0.0122]
    # L is actually 0.0997, but they didn't add up to 1
    temp_AAs_2 = 'AG'
    probs_2 = [0.5158,0.4842]
    
    randomProtein = ''
    
    #2 or 3 Gs or As = 3*(0.0701+0.0658)^2*(1-0.0701-0.0658) + (0.0701+0.0658)^3 = 0.0504
    #1 G or A = 0.3044
    #No G or A = 0.6452
    #3 Gs or As = (0.0701+0.0658)^3 = 0.00251
    
    for numNonrandomQs in range(1000):
        #randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        #randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        #randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        #randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        #randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        #randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        #randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        #randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        #randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        #randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        randomProtein = randomProtein + temp_AAs_2[np.random.choice(np.arange(0,2), p=probs_2)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
        randomProtein = randomProtein + temp_AAs[np.random.choice(np.arange(0,20), p=probs)]
    
    return randomProtein

# apply NN to 100x proteins each with 1000 potential cut sites centered at Qs. Print number that are and are not cut
def test_random_proteins(NN):
    cuts_count = 0
    not_cuts_count = 0
    for i in tqdm(range(10)):
        temp_random_protein = generate_random_protein()
        for j in range(3000):
            temp_vector = encode(temp_random_protein[10*j : 10*j+10])
            
            score = []
            for i, single_NN in enumerate(NN):
                score.append(round(NN[i].predict_proba([temp_vector]).flatten()[1], 0))
            avg_score = round(sum(score)/len(score),0)
            
            #print(temp_random_protein[10*j : 10*j+10], avg_score, score)
            
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

def make_seq_logo(large_truePosList, pseudocount):
    # make 10x20 PFM
    tempSeqLogo1 = np.zeros((10, 20))
    for i, seq in enumerate(large_truePosList):
        for j, AA in enumerate(seq):
            if i != j:
                tempSeqLogo1[j][AAs.index(AA)] = tempSeqLogo1[j][AAs.index(AA)] + 1
    
    # normalize the PFM to a PWM with pseudocount
    tempSeqLogo2 = np.zeros((10,20))
    for i in range(10):
        for j in range(20):
            tempSeqLogo2[i][j] = (tempSeqLogo1[i][j] + pseudocount)/(sum(tempSeqLogo1[i]) + 20*pseudocount)
    
    return tempSeqLogo2

def hyperparameter_optimization(large_uneven_inputs, large_uneven_outputs):
    for i in range(10):
        for split in [0.2]:#[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
            train_input, test_input, train_output, test_output = train_test_split(large_uneven_inputs, large_uneven_outputs, test_size=split, random_state=np.random.randint(10000))
            #print("Train =", sum(train_output), len(train_output))
            #print("Test =", sum(test_output), len(test_output))
            for solver in ['adam']:#, 'lbfgs', 'sgd']:
                for activation in ['relu']:#, 'identity', 'logistic', 'tanh']:
                    for num_CV in [10]:#[2,3,4,5,6,7,8,9,10,15,20,25]:
                        for os in ["None"]:#,"Naive","SMOTE","ADASYN","BorderlineSMOTE","SVMSMOTE"]:#,"KMeansSMOTE"]:
                           for HLs in [100]:#[2,3,4,5,10,20,50,100,200,500,1000]:#(10,2)]:
                                for a in [0.000001]:#[0.000000000001,0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1]:
                                    NN, CV_MCC = train_NN(train_input, train_output, HLs, a, os, num_CV, activation, solver)
                                    final_MCC_train = test_NN(NN, train_input, train_output)
                                    final_MCC_test = test_NN(NN, test_input, test_output)
                                    print(i, split, len(train_input), len(test_input), solver, activation, num_CV, os, HLs, a, "CV, 80% final, 20% final MCCs =", round(CV_MCC,5), round(final_MCC_train,5), round(final_MCC_test,5)) # note CV is on potentialy oversampled data, current 'None' oversampling
                                    
                                    #pickle.dump(NN, open(PATH + "NN_final_SarbeCoV-only_CV_MCC_" + str(round(CV_MCC,5)) + "_" + str(i) + ".sav", 'wb'))

def ensemble_NN_test(large_uneven_inputs, large_uneven_outputs):
    train_input, test_input, train_output, test_output = train_test_split(large_uneven_inputs, large_uneven_outputs, test_size=0.2, random_state=np.random.randint(10000))
    NNs = []
    for i in range(5):
        NNs.append(MLPClassifier(solver='adam', activation='tanh', alpha=0.00001, hidden_layer_sizes=3, random_state=np.random.randint(10000), max_iter=10000, learning_rate='invscaling', verbose=False))
        NNs[i].fit(train_input, train_output)
        print(i)
    
    train_predicted = [NNs[i].predict(train_input).tolist() for i in range(len(NNs))]
    test_predicted = [NNs[i].predict(test_input).tolist() for i in range(len(NNs))]
    
    NN_similarity_matrix = np.zeros([len(NNs), len(NNs)])
    train_cb, train_ciij, train_iicj, train_ib = np.zeros([len(NNs), len(NNs)]), np.zeros([len(NNs), len(NNs)]), np.zeros([len(NNs), len(NNs)]), np.zeros([len(NNs), len(NNs)])
    test_cb,  test_ciij,  test_iicj,  test_ib  = np.zeros([len(NNs), len(NNs)]), np.zeros([len(NNs), len(NNs)]), np.zeros([len(NNs), len(NNs)]), np.zeros([len(NNs), len(NNs)])
    
    for i in range(len(NNs)):
        for j in range(len(NNs)):
            temp_1 = NNs[i].coefs_[0].flatten().tolist() + NNs[i].coefs_[1].flatten().tolist()
            temp_2 = NNs[j].coefs_[0].flatten().tolist() + NNs[j].coefs_[1].flatten().tolist()
            NN_similarity_matrix[i][j] = cosine_similarity([temp_1], [temp_2])
                                    
            for k in range(len(train_output)):
                if train_predicted[i][k] == train_output[k] and train_predicted[j][k] == train_output[k]:
                    train_cb[i][j] += 1
                elif train_predicted[i][k] == train_output[k] and train_predicted[j][k] != train_output[k]:
                    train_ciij[i][j] += 1
                elif train_predicted[i][k] != train_output[k] and train_predicted[j][k] == train_output[k]:
                    train_iicj[i][j] += 1
                elif train_predicted[i][k] != train_output[k] and train_predicted[j][k] != train_output[k]:
                    train_ib[i][j] += 1
            
            for k in range(len(test_output)):
                if test_predicted[i][k] == test_output[k] and test_predicted[j][k] == test_output[k]:
                    test_cb[i][j] += 1
                elif test_predicted[i][k] == test_output[k] and test_predicted[j][k] != test_output[k]:
                    test_ciij[i][j] += 1
                elif test_predicted[i][k] != test_output[k] and test_predicted[j][k] == test_output[k]:
                    test_iicj[i][j] += 1
                elif test_predicted[i][k] != test_output[k] and test_predicted[j][k] != test_output[k]:
                    test_ib[i][j] += 1
    
    train_ci_similarity_matrix = np.zeros([len(NNs), len(NNs)])
    test_ci_similarity_matrix  = np.zeros([len(NNs), len(NNs)])
    for i in range(len(NNs)):
        for j in range(len(NNs)):
            train_ci_similarity_matrix[i][j] = train_ib[i][j] / (train_ib[i][j] + train_ciij[i][j] + train_iicj[i][j])
            test_ci_similarity_matrix[i][j]  = test_ib[i][j]  / (test_ib[i][j]  + test_ciij[i][j]  + test_iicj[i][j])
    
    #train_tn, train_fp, train_fn, train_tp = 0, 0, 0, 0
    #test_tn,  test_fp,  test_fn,  test_tp  = 0, 0, 0, 0
    
    #test_tn,  test_fp,  test_fn,  test_tp  = confusion_matrix(train_output, NN.predict(train_input)).ravel()
    #train_tn, train_fp, train_fn, train_tp = confusion_matrix(test_output,  NN.predict(test_input )).ravel()
    
    #print([train_tn, train_fp, train_fn, train_tp], [test_tn,  test_fp,  test_fn,  test_tp])
    
    print(NN_similarity_matrix)
    
    print(train_cb)
    print(train_ciij)
    print(train_iicj)
    print(train_ib)
    
    print(test_cb)
    print(test_ciij)
    print(test_iicj)
    print(test_ib)
    
    print(train_ci_similarity_matrix)
    print(test_ci_similarity_matrix)

def taxonomy_subset(large_trueNegList, large_trueNegSubgenusList, large_truePosList, large_truePosSubgenusList, subset_term):
    subset_trueNegList = []
    excluded_trueNegList = []
    subset_truePosList = []
    excluded_truePosList = []
    if subset_term == "AlphaCoV":
        for i, neg in enumerate(large_trueNegList):
            if "ColaCoV" in large_trueNegSubgenusList[i] or "DecaCoV" in large_trueNegSubgenusList[i] or "DuvinaCoV" in large_trueNegSubgenusList[i] or "LuchaCoV" in large_trueNegSubgenusList[i] or "MinaCoV" in large_trueNegSubgenusList[i] or "MinunaCoV" in large_trueNegSubgenusList[i] or "MyotaCoV" in large_trueNegSubgenusList[i] or "NyctaCoV" in large_trueNegSubgenusList[i] or "PedaCoV" in large_trueNegSubgenusList[i] or "RhinaCoV" in large_trueNegSubgenusList[i] or "SetraCoV" in large_trueNegSubgenusList[i] or "SunaCoV" in large_trueNegSubgenusList[i] or "TegaCoV" in large_trueNegSubgenusList[i]:
                subset_trueNegList.append(neg)
            else:
                excluded_trueNegList.append(neg)
        for i, pos in enumerate(large_truePosList):
            if "ColaCoV" in large_truePosSubgenusList[i] or "DecaCoV" in large_truePosSubgenusList[i] or "DuvinaCoV" in large_truePosSubgenusList[i] or "LuchaCoV" in large_truePosSubgenusList[i] or "MinaCoV" in large_truePosSubgenusList[i] or "MinunaCoV" in large_truePosSubgenusList[i] or "MyotaCoV" in large_truePosSubgenusList[i] or "NyctaCoV" in large_truePosSubgenusList[i] or "PedaCoV" in large_truePosSubgenusList[i] or "RhinaCoV" in large_truePosSubgenusList[i] or "SetraCoV" in large_truePosSubgenusList[i] or "SunaCoV" in large_truePosSubgenusList[i] or "TegaCoV" in large_truePosSubgenusList[i]:
                subset_truePosList.append(pos)
            else:
                excluded_truePosList.append(pos)
    elif subset_term == "BetaCoV":
        for i, neg in enumerate(large_trueNegList):
            if "EmbeCoV" in large_trueNegSubgenusList[i] or "HibeCoV" in large_trueNegSubgenusList[i] or "MerbeCoV" in large_trueNegSubgenusList[i] or "NobeCoV" in large_trueNegSubgenusList[i] or "SarbeCoV" in large_trueNegSubgenusList[i]:
                subset_trueNegList.append(neg)
            else:
                excluded_trueNegList.append(neg)
        for i, pos in enumerate(large_truePosList):
            if "EmbeCoV" in large_truePosSubgenusList[i] or "HibeCoV" in large_truePosSubgenusList[i] or "MerbeCoV" in large_truePosSubgenusList[i] or "NobeCoV" in large_truePosSubgenusList[i] or "SarbeCoV" in large_truePosSubgenusList[i]:
                subset_truePosList.append(pos)
            else:
                excluded_truePosList.append(pos)
    elif subset_term == "GammaCoV":
        for i, neg in enumerate(large_trueNegList):
            if "BrangaCoV" in large_trueNegSubgenusList[i] or "CegaCoV" in large_trueNegSubgenusList[i] or "IgaCoV" in large_trueNegSubgenusList[i]:
                subset_trueNegList.append(neg)
            else:
                excluded_trueNegList.append(neg)
        for i, pos in enumerate(large_truePosList):
            if "BrangaCoV" in large_truePosSubgenusList[i] or "CegaCoV" in large_truePosSubgenusList[i] or "IgaCoV" in large_truePosSubgenusList[i]:
                subset_truePosList.append(pos)
            else:
                excluded_truePosList.append(pos)
    elif subset_term == "DeltaCoV":
        for i, neg in enumerate(large_trueNegList):
            if "AndeCoV" in large_trueNegSubgenusList[i] or "BuldeCoV" in large_trueNegSubgenusList[i] or "HerdeCoV" in large_trueNegSubgenusList[i]:
                subset_trueNegList.append(neg)
            else:
                excluded_trueNegList.append(neg)
        for i, pos in enumerate(large_truePosList):
            if "AndeCoV" in large_truePosSubgenusList[i] or "BuldeCoV" in large_truePosSubgenusList[i] or "HerdeCoV" in large_truePosSubgenusList[i]:
                subset_truePosList.append(pos)
            else:
                excluded_truePosList.append(pos)
    elif subset_term in ["ColaCoV", "DecaCoV", "DuvinaCoV", "LuchaCoV", "MinaCoV", "MinunaCoV", "MyotaCoV", "NyctaCoV", "PedaCoV", "RhinaCoV", "SetraCoV", "SunaCoV", "TegaCoV","EmbeCoV", "HibeCoV", "MerbeCoV", "NobeCoV", "SarbeCoV","BrangaCoV", "CegaCoV", "IgaCoV","AndeCoV", "BuldeCoV", "HerdeCoV","Unclassified"]:
        for i, neg in enumerate(large_trueNegList):
            if subset_term in large_trueNegSubgenusList[i]:
                subset_trueNegList.append(neg)
            else:
                excluded_trueNegList.append(neg)
        for i, pos in enumerate(large_truePosList):
            if subset_term in large_truePosSubgenusList[i]:
                subset_truePosList.append(pos)
            else:
                excluded_truePosList.append(pos)
    else: # for lists of subgenera
        for i, neg in enumerate(large_trueNegList):
            if not set(subset_term).isdisjoint(large_trueNegSubgenusList[i].split(",")):#bool(set(subset_term) & set([large_trueNegSubgenusList[i]])):
                subset_trueNegList.append(neg)
            else:
                excluded_trueNegList.append(neg)
        for i, pos in enumerate(large_truePosList):
            if not set(subset_term).isdisjoint(large_truePosSubgenusList[i].split(",")):#bool(set(subset_term) & set([large_truePosSubgenusList[i]])):
                subset_truePosList.append(pos)
            else:
                excluded_truePosList.append(pos)
    
    return subset_trueNegList, excluded_trueNegList, subset_truePosList, excluded_truePosList

def main():
    # read files [human_proteome, CoV 4268x positives (mainly Qs), CoV negative 17493x Qs and 11421x Hs]
    comIDList, comSeqList, large_trueNegList, large_trueNegSubgenusList, large_truePosList, large_truePosSubgenusList = read_files(human_proteome_file_name, large_trueNegFileName, large_truePosFileName)
    print("Human proteome size =", len(comIDList))
    print("Pos and neg size =", len(large_truePosList), len(large_trueNegList))
    
    #subgenera_order = ['SarbeCoV','HibeCoV','NobeCoV','MerbeCoV','EmbeCoV','MyotaCoV','DecaCoV','NyctaCoV','DuvinaCoV','ColaCoV','SunaCoV','SetraCoV','MinunaCoV','PedaCoV','RhinaCoV','BrangaCoV','IgaCoV','MinaCoV','LuchaCoV','CegaCoV','TegaCoV','HerdeCoV','BuldeCoV','AndeCoV','Unclassified']
    #for iteration in range(10):
    #    train_subset_unevenInputList_2D = []
    #    train_subset_unevenOutputList_2D = []
    #    test_subset_unevenInputList_2D = []
    #    test_subset_unevenOutputList_2D = []
    #    for subgenus in subgenera_order:
    #        temp_subset_trueNegList, _, temp_subset_truePosList, _ = taxonomy_subset(large_trueNegList, large_trueNegSubgenusList, large_truePosList, large_truePosSubgenusList, subgenus)
    #        large_unevenInputList, large_unevenOutputList = encode_list_one_hot(temp_subset_trueNegList, temp_subset_truePosList)
    #        print(subgenus, len(temp_subset_trueNegList), len(temp_subset_truePosList))
    #        temp_train_input, temp_test_input, temp_train_output, temp_test_output = train_test_split(large_unevenInputList, large_unevenOutputList, test_size=0.2, random_state=np.random.randint(10000))
    #        train_subset_unevenInputList_2D.append(temp_train_input)
    #        train_subset_unevenOutputList_2D.append(temp_train_output)
    #        test_subset_unevenInputList_2D.append(temp_test_input)
    #        test_subset_unevenOutputList_2D.append(temp_test_output)
    #    for i in range(len(subgenera_order)):
    #        train_input = [temp_1 for temp_0 in train_subset_unevenInputList_2D[0:i+1] for temp_1 in temp_0]
    #        train_output = [temp_1 for temp_0 in train_subset_unevenOutputList_2D[0:i+1] for temp_1 in temp_0]            
    #        NN, CV_MCC = train_NN(train_input, train_output, 10, 0.00001, 'None', 5, 'relu', 'lbfgs')
    #        subgenus_MCCs = []
    #        for j in range(len(subgenera_order)):
    #            if j <= i:
    #                subgenus_MCCs.append(test_NN(NN, test_subset_unevenInputList_2D[j], test_subset_unevenOutputList_2D[j]))
    #            else:
    #                subgenus_MCCs.append(test_NN(NN, train_subset_unevenInputList_2D[j]+test_subset_unevenInputList_2D[j], train_subset_unevenOutputList_2D[j]+test_subset_unevenOutputList_2D[j]))
    #        print(iteration, i, round(CV_MCC,5), subgenus_MCCs)#, round(final_MCC_train,5))
    
    
    
    #for subgenus in ["AlphaCoV", "BetaCoV", "GammaCoV", "DeltaCoV"]:
    #for subgenus in ["ColaCoV", "DecaCoV", "DuvinaCoV", "LuchaCoV", "MinaCoV", "MinunaCoV", "MyotaCoV", "NyctaCoV", "PedaCoV", "RhinaCoV", "SetraCoV", "SunaCoV", "TegaCoV","EmbeCoV", "HibeCoV", "MerbeCoV", "NobeCoV", "SarbeCoV","BrangaCoV", "CegaCoV", "IgaCoV","AndeCoV", "BuldeCoV", "HerdeCoV"]:
    #    subset_trueNegList, excluded_trueNegList, subset_truePosList, excluded_truePosList = taxonomy_subset(large_trueNegList, large_trueNegSubgenusList, large_truePosList, large_truePosSubgenusList, subgenus)
    #    large_unevenSubsetInputList, large_unevenSubsetOutputList = encode_list_one_hot(subset_trueNegList, subset_truePosList)
    #    large_unevenExcludedInputList, large_unevenExcludedOutputList = encode_list_one_hot(excluded_trueNegList, excluded_truePosList)
    #    
    #    for i in range(3):
    #        train_input, test_input, train_output, test_output = train_test_split(large_unevenExcludedInputList, large_unevenExcludedOutputList, test_size=0.2, random_state=np.random.randint(10000))
    #        NN, CV_MCC = train_NN(train_input, train_output, 100, 0.00001, 'None', 0, 'relu', 'lbfgs')
    #        final_MCC_train = test_NN(NN, train_input, train_output)
    #        final_MCC_test = test_NN(NN, test_input, test_output)
    #        final_MCC_excluded_test = test_NN(NN, large_unevenSubsetInputList, large_unevenSubsetOutputList)
    #        print(subgenus, i, int(sum(large_unevenSubsetOutputList)), int(len(large_unevenSubsetOutputList)-sum(large_unevenSubsetOutputList)), int(sum(large_unevenExcludedOutputList)), int(len(large_unevenExcludedOutputList)-sum(large_unevenExcludedOutputList)), round(CV_MCC,5), round(final_MCC_train,5), round(final_MCC_test,5), round(final_MCC_excluded_test,5))
    
    
    #large_trueNegList, _, large_truePosList, _ = taxonomy_subset(large_trueNegList, large_trueNegSubgenusList, large_truePosList, large_truePosSubgenusList, 'SarbeCoV')
    #print("SarbeCoV-specific pos and neg size =", len(large_truePosList), len(large_trueNegList))
    #large_trueNegList, _, large_truePosList, _ = taxonomy_subset(large_trueNegList, large_trueNegSubgenusList, large_truePosList, large_truePosSubgenusList, 'BetaCoV')
    #print("BetaCoV-specific pos and neg size =", len(large_truePosList), len(large_trueNegList))
    
    # encode CoV inuts as one-hot, 4x physiochem parameters, or positive and/or negative sequence logo probabilities (with small but nonzero pseudocount = 0.01)
    ###large_unevenInputList, large_unevenOutputList = encode_list_one_hot(large_trueNegList, large_truePosList)
    #large_unevenInputList, large_unevenOutputList = encode_list_physiochem(large_trueNegList, large_truePosList)
    #large_unevenInputList, large_unevenOutputList = encode_list_pos_seq_logo(large_trueNegList, large_truePosList, 0.01)
    #large_unevenInputList, large_unevenOutputList = encode_list_both_seq_logos(large_trueNegList, large_truePosList, 0.01)
    
    # optimize split frac, solver, activation func, num_CV, oversampling, HLs, and regularization
    ###hyperparameter_optimization(large_unevenInputList, large_unevenOutputList)
    
    
    
    #ensemble_NN_test(large_unevenInputList, large_unevenOutputList)
    
    #seq_logo = make_seq_logo(large_truePosList, 0.01)
    
    ###NN, MCC, tn, fp, fn, tp = train_NN(large_unevenInputList, large_unevenOutputList, 10, 0.00001, 'Nonw', 10, 'tanh', 'adam')
    
    #NN = MLPClassifier(solver='adam', activation='tanh', alpha=0.00001, hidden_layer_sizes=10, random_state=np.random.randint(10000), max_iter=10000, learning_rate='invscaling', verbose=False)
    #NN.fit(large_unevenInputList, large_unevenOutputList)
    
    # Make histogram for positive seq logo scores
    #scores_pos = np.zeros(100)
    #for seq in tqdm(large_truePosList):
    #    temp = int(np.dot(encode(seq), [np.log(i) for i in np.array(seq_logo).flatten()]))
    #    scores_pos[temp] += 1
    #print(scores_pos)
    
    # Make histogram for negative seq logo scores
    #scores_neg = np.zeros(100)
    #for seq in tqdm(large_trueNegList):#random.sample(large_trueNegList,100000)):
    #    temp = int(np.dot(encode(seq), [np.log(i) for i in np.array(seq_logo).flatten()]))
    #    scores_neg[temp] += 1
    #print(scores_neg)
    
    # Undersample large_trueNegList based on seq logo score. Only log scores >=-30 are included
    #large_trueNegList_US = []
    #for seq in tqdm(large_trueNegList):
    #    temp = int(np.dot(encode(seq), [np.log(i) for i in np.array(seq_logo).flatten()]))
    #    if temp >= -25:
    #        large_trueNegList_US.append(seq)
    #print(len(large_trueNegList_US))
    
    #pickle.dump(large_trueNegList_US, open(PATH + "large_trueNegList_US_-25.sav", 'wb'))
    
    
    
    NNs = []
    for i in range(10):
        large_trueNegList_US = random.sample(large_trueNegList, len(large_truePosList))
        #print("US neg list & pos list lens =", len(large_trueNegList_US), len(large_truePosList))
        
        # encode CoV inuts as one-hot, 4x physiochem parameters, or positive and/or negative sequence logo probabilities (with small but nonzero pseudocount = 0.01)
        large_unevenInputList, large_unevenOutputList = encode_list_one_hot(large_trueNegList_US, large_truePosList)
        #large_unevenInputList, large_unevenOutputList = encode_list_physiochem(large_trueNegList, large_truePosList)
        #large_unevenInputList, large_unevenOutputList = encode_list_pos_seq_logo(large_trueNegList, large_truePosList, 0.01)
        #large_unevenInputList, large_unevenOutputList = encode_list_both_seq_logos(large_trueNegList, large_truePosList, 0.01)
        
        # hyperparameter optimization many nested for loops
        #hyperparameter_optimization(large_unevenInputList, large_unevenOutputList)
        
        train_input, test_input, train_output, test_output = train_test_split(large_unevenInputList, large_unevenOutputList, test_size=0.2, random_state=np.random.randint(10000))
        NN, CV_MCC = train_NN(train_input, train_output, test_input, test_output, 2, 0.00001, 'None', 3, 'tanh', 'lbfgs')
        test_MCC, confusion = test_NN(NN, test_input, test_output)
        print("CV & 20% test MCCs, 20% test confusion matrix (TN, FP, FN, TP) =", round(CV_MCC,5), round(test_MCC,5), np.array(confusion).flatten())
        NNs.append(NN)
    
    # or load saved NN to avoid retraining cycles
    #loaded_NN = pickle.load(open(PATH + "3.2.1 NN_final_all-CoV_CV_MCC_0.91462_0.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.2 NN_final_all-CoV_CV_MCC_0.89904_1.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.3 NN_final_all-CoV_CV_MCC_0.91253_2.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.4 NN_final_BetaCoV-only_CV_MCC_0.96330_0.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.5 NN_final_BetaCoV-only_CV_MCC_0.95624_1.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.6 NN_final_BetaCoV-only_CV_MCC_0.92631_2.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.7 NN_final_SarbeCoV-only_CV_MCC_0.92859_0.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.8 NN_final_SarbeCoV-only_CV_MCC_0.94006_1.sav", "rb"))
    #loaded_NN = pickle.load(open(PATH + "3.2.9 NN_final_SarbeCoV-only_CV_MCC_0.98265_2.sav", "rb"))
    
    #final_MCC = test_NN(loaded_NN, large_unevenInputList, large_unevenOutputList)
    #print("Final MCC =", final_MCC)
    
    # test trained or loaded NN(s) on 20350 human proteins
    #print("Testing NN(s) on human proteome")
    #test_human_proteome(loaded_NN, comIDList, comSeqList, "Human_prediction___NN_final_all-CoV_0.csv")#human_proteome_prediction_file_name)
    
    # test trained or loaded NN(s) on 100,000x Gs or As to estimate random cleavage rate
    #print("testing NN(s) on random proteins")
    #test_random_proteins(loaded_NN)



main()









