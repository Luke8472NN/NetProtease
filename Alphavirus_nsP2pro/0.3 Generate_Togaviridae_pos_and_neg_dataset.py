from tqdm import tqdm
import csv

PATH = "C:\\Users\\TEMP\\Downloads\\"
CoV_pp1as_file_name = "Togaviridae_1797x_for_negs.fasta"

positives = ['ADIGAALVET','ADTGAALVET','ADVGAALVET','ARAGAGVVNT','DDAGAALVET','DDIGAALVDT','DDIGAALVET','DDIGGALVET','DDVGAALVET','DDVGGALVET','DGAGATIIDC','DGPGATIIDT','DGVGATIIDC','DGVGATLVDC','DGVGSTIIDC','DRAGAGIIEN','DRAGAGIIET','DRAGAGIVET','DRAGAGMIET','DRAGAGTIET','EQPGAGYIET','ERAGAGVVET','FRAGAGVVET','GRAGAGIIET','LGTGATVALK','NDIGAALVET','QEAGAGSAET','QEAGAGSVET','QEAGAGTVET','QEEGAGSVET','QRAGEGVVET','TDIGAALVET','YHAGAGVVET','YRAGAGVVET','DDVGAAPSYT','DGSGAAPSYR','DGVASAPAYR','DGVGAAPAYR','DGVGAAPSYK','DGVGAAPSYR','DGVGAAPSYS','DGVGAAPSYT','EFAGAAPSYD','EGVGAAPSYR','HEAGCAPSYH','HEAGRAPAYR','HEAGSAPSYH','HEAGTAPSYH','HTAGCAPSYR','IEIGAAPSYT','LPAGNAPAYR','LPAGSAPAYR','MTAGCAPAYR','PEVGSAPTYR','PMVGSAPTYC','PQVGAAARYR','PRAGAAPAYR','QAAGCAPAYA','QAAGCAPAYT','QAAGCAPVYA','QPAGSAPMYT','QPAGTAPNYR','SMVGAAPGYR','TGIGCAPSYR','TRAGCAPSNK','TRAGCAPSYR','TRTGCAPSYR','YEAGRAPAYR','ARAGAYIFSS','DGAGAYIFSS','DGAGSYIFSS','DGLGGYIFSS','DGPGGYIFSS','DRAGAYIFSS','DRAGGYIFSS','DRAGGYTFSS','ERAGAYIFSS','ERPGGYIFSS','FDAGAYIFSS','FDAGAYTFSS','FEAGAYIFSS','GRAGAYIFSS','GRAGAYIFST','GRAGGYIFSS','LGVGAYIFSS','NGVGGYIFST','SRAGAYIFSS','TGAGGYIFSS','TGVGGFIFSS','TGVGGYIFSS','TGVGGYIFST','YDAGAYIFSS','YEAGAYIFSS']

def read_CoV_fasta(CoV_pp1as_file_name):
    # read human proteome fasta
    com = open(PATH + CoV_pp1as_file_name, 'r')
    comIDList = []
    comSeqList = []
    tempSeq = ''
    for i, line in enumerate(com):
        if line.split(' ')[0][0] == '>':
            if tempSeq != '':
                comSeqList.append(tempSeq)
                tempSeq = ''
            comIDList.append(line.split(' ')[0])
        else:
            tempSeq = tempSeq + line.replace("\n", "")
    comSeqList.append(tempSeq) #add one last seq
    
    return comIDList, comSeqList



def main():
    CoV_IDs, CoV_pp1as = read_CoV_fasta(CoV_pp1as_file_name)
    
    with open(PATH + 'Togaviridae_negatives_no_repeats.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        negatives = []
        for i, pp1a in enumerate(tqdm(CoV_pp1as)):
            for j in range(len(pp1a)-10):
                if pp1a[j+3] == "G" and "X" not in pp1a[j:j+10] and "B" not in pp1a[j:j+10] and "J" not in pp1a[j:j+10] and "Z" not in pp1a[j:j+10] and pp1a[j:j+10] not in positives and pp1a[j:j+10] not in negatives:
                    negatives.append(pp1a[j:j+10])
                    writer.writerow([pp1a[j:j+10]])
                    #print(pp1a[j:j+10])
    
    print(len(positives), len(negatives))
    print(len(CoV_IDs), len(CoV_pp1as))

main()