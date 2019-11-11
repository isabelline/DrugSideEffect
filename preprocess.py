from sklearn.model_selection import KFold
import numpy as np

with open("/user_home/hajung/AdverseDrugReaction/ADE-Corpus-V2/DRUG-DOSE.rel", 'r',encoding='ascii') as f:
    lines = f.readlines()

ListString2ddLines = [x.split("|")[1] for x in lines]
ListInt2ddLabels = [1] * len(ListString2ddLines)

with open("/user_home/hajung/AdverseDrugReaction/ADE-Corpus-V2/DRUG-AE.rel", 'r',encoding='ascii') as f:
    lines = f.readlines()

ListString2daLines = [x.split("|")[1] for x in lines]
ListInt2daLabels = [0] * len(ListString2daLines)

with open("/user_home/hajung/AdverseDrugReaction/ADE-Corpus-V2/ADE-NEG.txt", 'r',encoding='ascii') as f:
    lines = f.readlines()

ListString2negLines = [" ".join(x.split(" ")[2:]) for x in lines]
ListInt2negLabels = [2] * len(ListString2negLines)

print("DD size: " + str(len(ListInt2ddLabels)))
print("DA size: " + str(len(ListInt2daLabels)))
print("NEG size: " + str(len(ListInt2negLabels)))

ListString2allLines = np.array(ListString2daLines + ListString2ddLines + ListString2negLines)

ListInt2allLabels = np.array(ListInt2daLabels + ListInt2ddLabels + ListInt2negLabels)

indices = np.arange(ListString2allLines.shape[0])
np.random.shuffle(indices)
ListString2allLines = ListString2allLines[indices]
ListInt2allLabels = ListInt2allLabels[indices]


kf = KFold(n_splits=10, shuffle=False)

cnt = 1
for train_index, test_index in kf.split(ListString2allLines):
    X_train, X_test = ListString2allLines[train_index], ListString2allLines[test_index]
    y_train, y_test = ListInt2allLabels[train_index], ListInt2allLabels[test_index]
    with open("./CV/train_"+str(cnt)+".tsv", 'w', encoding='utf-8') as f:
        for i,line in enumerate(X_train):
            f.write(line.strip() + "\t"+ str(y_train[i])+"\n")
    
    with open("./CV/test_"+str(cnt)+".tsv", 'w', encoding='utf-8') as f:
        for i,line in enumerate(X_test):
            f.write(line.strip() + "\t"+ str(y_test[i])+"\n")
    cnt += 1
