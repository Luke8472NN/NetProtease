import numpy as np
from sklearn.manifold import TSNE
import csv

PATH = "C:\\Users\\lprescott\\Documents\\Human proteome\\3CLpro GitHub\\"
whole_pp1ab_file_name = "1.1 GenBank_CoV_pp1ab_4161x.fasta"
blosum62_and_cuts_file_name = "2.4 BLOSUM62_matrix_and_388x11_cleavages.csv"

blosum_ordering = "ARNDCQEGHILKMFPSTWYV"



# read all 4161x CoV pp1abs and file with 
def read_files(blosum62_and_cuts_file_name, whole_pp1ab_file_name):
    blosum62 = []
    species_list = []
    cuts_list = []
    
    file = open(PATH + blosum62_and_cuts_file_name, 'r')
    for i, line in enumerate(file):
        if i>=1 and i<=20:
            blosum62.append([int(element) for element in line.replace("\n","").split(",")[1:21]])
        if i>=22:
            species_list.append(line.split(",")[0])
            cuts_list.append(line.replace("\n","").split(",")[1:12])
    file.close()
    
    
    species_list2 = []
    pp1ab_list = []
    
    file = open(PATH + whole_pp1ab_file_name, 'r')
    species_list2 = []
    pp1ab_list = []
    tempSeq = ''
    for i, line in enumerate(file):
        if line.split(' ')[0][0] == '>':
            if tempSeq != '':
                pp1ab_list.append(tempSeq)
                tempSeq = ''
            species_list2.append(line.split(' ')[0])
        else:
            tempSeq = tempSeq + line.replace("\n", "")
    pp1ab_list.append(tempSeq) #add one last seq
    file.close()
    
    
    return blosum62, species_list, cuts_list, species_list2, pp1ab_list

protease_list = []

def find_3CLpros_in_pp1abs():
    for i, species1 in enumerate(species_list):
        for j, species2 in enumerate(species_list2):
            if species1 == species2:
                protease_list.append(pp1ab_list[j][pp1ab_list[j].index(cuts_list[i][0])+5 : pp1ab_list[j].index(cuts_list[i][1])+5])
                print(protease_list[len(protease_list)-1])
                break
    
    #with open(PATH + 'All_CoV_3CLpros_4161x.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    for i, species1 in enumerate(species_list):
    #        writer.writerow([species1, protease_list[i]])

# read files
blosum62, species_list, cuts_list, species_list2, pp1ab_list = read_files(blosum62_and_cuts_file_name, whole_pp1ab_file_name)

print("# cut sets =", len(species_list)) #len(cuts_list)
print("# pp1abs =", len(species_list2)) #len(pp1ab_list)

# find and write to a file att 4161x 3CLpro seqs between cuts 1 and 2
find_3CLpros_in_pp1abs()






