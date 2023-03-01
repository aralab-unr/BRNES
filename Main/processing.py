# you need to run BRNES.py, AdhocTD.py, and DARL.py for multiple environment before running this file.

import sys
import math
import pickle
import numpy as np
import pandas as pd

episodeNum = int(sys.argv[1])
sg_gap = int(sys.argv[2])
RewardGap = int(sys.argv[3])
convGap = int(sys.argv[4])

#### Edit these lines according to your experiments and Graph generation preferences #################
# specifies framework names
frameworkName = ['BRNES','AdhocTD','DARL'] 

# look into 'Main/OutputFile/BRNES.txt to fetch your desired file name' 
# place those file names sequentially (BRNES-AdhocTD-DARL) inside fName variable. 
# here, '1729b'/a2075'/'dc756' are some example file names

# fName = [
#         ['f0946', 'e2365', '77acb'],
#         ['cb255', 'a2075', '177c9'],
#         ['56762', 'dc756', 'cc973'],
#         ['56762', 'dc756', 'cc973']
#         ]

fName = [
        ['f0946', 'e2365', '77acb'],
        ['f0946', 'e2365', '77acb'],
        ['f0946', 'e2365', '77acb'],
        ['f0946', 'e2365', '77acb']
        ]

# fName = [
#         ['3761b', 'd11b6', 'a6e1f'],
#         ['3761b', 'd11b6', 'a6e1f'],
#         ['3761b', 'd11b6', 'a6e1f'],
#         ['3761b', 'd11b6', 'a6e1f']
#         ]

# specify attackers' percentage here
AttackPercentage=['0','10', '30', '40']
######################################################################################################

BRNESList = []
AdhocTDList = []
DARLList = []
df = pd.DataFrame({})
df1 = pd.DataFrame({})
df2 = pd.DataFrame({})
df3 = pd.DataFrame({})

for i in range(len(fName)):
    for j in range(len(frameworkName)):
        fileName = fName[i][j]
        
        ###### SG ####################################################################################
        with open("./SG/"+str(fileName)+"_"+str(frameworkName[j])+"_Step", "rb") as Sp:
            stepVal = pickle.load(Sp)
        
        stepValAvg = [np.average(x) for x in zip(*stepVal)]
        stepValAvg = [stepValAvg[x:x+sg_gap] for x in range(0, len(stepValAvg),sg_gap)]
        stepValAvg = np.mean(np.array(stepValAvg), axis = 1)
        
        ###### Reward ####################################################################################
        with open("./Reward/"+str(fileName)+"_"+str(frameworkName[j])+"_Reward", "rb") as Rp:
            rewardVal = pickle.load(Rp)
        epochAvg = []
        for y in range(len(rewardVal[0])):
            epochAvg.append([np.average(z) for z in zip(*(rewardVal[x][y] for x in range(len(rewardVal))))])
            
        rewardValAvg = [list(x) for x in zip(*epochAvg)]
        listItem = []
        listItemAvg = []
        
        for p in range(episodeNum):
            for item in rewardValAvg:
                listItem.append(item[p])
            listItemAvg.append(sum(listItem)/len(listItem))
            listItem = []
            
        rewardValAvg0 = [listItemAvg[x:x+RewardGap] for x in range(0, len(listItemAvg),RewardGap)]
        rewardValAvg0 = np.mean(np.array(rewardValAvg0), axis = 1)
        
        
        
        ###### Convergence  ####################################################################################
        with open("./Convergence/"+str(fileName)+"_"+str(frameworkName[j])+"_convergence", "rb") as Cp:
            convergenceVal = pickle.load(Cp)
        convergenceAvg = [np.average(x) for x in zip(*convergenceVal)]
        convergenceAvg = [convergenceAvg[x:x+convGap] for x in range(0, len(convergenceAvg),convGap)]
        convergenceAvg = np.mean(np.array(convergenceAvg), axis = 1)
        
        
        
        # ###### TG ####################################################################################
        with open("./TG/"+str(fileName)+"_"+str(frameworkName[j])+"_Time", "rb") as Tp:
            timeVal = pickle.load(Tp)
        
        if frameworkName[j]=='BRNES':
            df3["BRNES"+str(AttackPercentage[i])]=[timeVal]
            df2["BRNES"+str(AttackPercentage[i])]=convergenceAvg
            df["BRNES"+str(AttackPercentage[i])]=stepValAvg
            df1["BRNES"+str(AttackPercentage[i])]=rewardValAvg0
        elif frameworkName[j]=='AdhocTD':
            df["AdhocTD"+str(AttackPercentage[i])]=stepValAvg
            df1["AdhocTD"+str(AttackPercentage[i])]=rewardValAvg0
            df2["AdhocTD"+str(AttackPercentage[i])]=convergenceAvg
            df3["AdhocTD"+str(AttackPercentage[i])]=[timeVal]
        elif frameworkName[j]=='DARL':
            df["DARL"+str(AttackPercentage[i])]=stepValAvg
            df1["DARL"+str(AttackPercentage[i])]=rewardValAvg0
            df2["DARL"+str(AttackPercentage[i])]=convergenceAvg
            df3["DARL"+str(AttackPercentage[i])]=timeVal
        else:
            print('*Error in processing file*')
            
df.to_csv("./ProcessedOutput/ProcessedSG.csv", index=False)
df1.to_csv("./ProcessedOutput/ProcessedReward.csv", index=False)
df2.to_csv("./ProcessedOutput/ProcessedConvergence.csv", index=False)
df3.to_csv("./ProcessedOutput/ProcessedTG.csv", index=False)
print("Processed files are ready inside the Main/ProcessedOutput folder")
        
            
        
        