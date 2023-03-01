#!/usr/bin/env python
# coding: utf-8
import sys
import os
from PP_environment.environment import Env
from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import random
import pickle
import uuid
import time
import math
import os
import copy
from operator import add, sub, mul

## starting main program
start = time.time()


gridHeightList = [int(sys.argv[1])]
gridWidthList = [int(sys.argv[1])]
noAgentList = [int(sys.argv[2])]
noObsList = [int(sys.argv[3])]
eList = [int(sys.argv[4])]
LoopVal = int(sys.argv[5]) # defines how many times the code will run
neighborWeightsList = [float(sys.argv[6])]
attackPercentage = [int(sys.argv[7])]
display = sys.argv[8]
sleep = float(sys.argv[9])
try:
    mode = sys.argv[10]
except:
    mode = 'random'
if mode.lower()=='random':
    playModeList = {"Agent":'random', "Target":'static', "Obstacle":'random', "Freeway":'random'}
else:
    playModeList = {"Agent":'random', "Target":'static', "Obstacle":'static', "Freeway":'static'}
flag = 1 # flag = 0, neighbor zone enabled and flag = 1, neighbor zone disabled

noTarget = 1 # there is only one target
noFreeway = 1 # there is only one freeway/resting area
AttackerList = [random.sample(range(0,noAgentList[0]), math.ceil(noAgentList[0]*attackPercentage[0]/100))] # calculating attackers' list
print("Attackers: ", AttackerList)

 # reward and penalties
actionReward = 0
obsReward = -1.5
freewayReward = 0.5
emptycellReward = 0
hitwallReward = -0.5
goalReward = 10

min_v = obsReward+hitwallReward
max_v = goalReward+freewayReward+emptycellReward

# hyper-parameters
alpha = 0.1 # RL learning rate
varepsilon = 1 # privacy budget


 # initializing lists for calculating and saving convergence values
diffAvg1 = []
diffAvg2 = []
diffAvg3 = []
diffAvg4 = []
diffAvg5 = []

# attack function for swapping the max Q-value with min Q-Value
def swap(a):
    max_index=a.index(max(a))
    min_index=a.index(min(a))
    ma=max(a)
    mi=min(a)
    a[max_index]=mi
    a[min_index]=ma
    return a

# Main Loop
for CriteriaVal in range(len(gridWidthList)):
    print("##################### Criteria Value: "+str(CriteriaVal)+" #######################\n")
    Attacker = AttackerList[CriteriaVal]
    Behb_tot = [100000 for i in range(noAgentList[CriteriaVal])] # advisee's budget for seeking advice during experience harvesting (EH)
    Besb_tot = [10000 for i in range(noAgentList[CriteriaVal])] # advisors' budget for seeking advice during experience giving (EG)
    fileName = str(uuid.uuid4())[:5] # initializing unique filename for storing learning outcomes
    stepsListFinal = []
    rewards_all_episodesFinal = []
    qtableListFinal = []
    diffAvg5 = []
    for countVal in range(LoopVal):
        gridWidth = gridWidthList[CriteriaVal]#10
        gridHeight = gridHeightList[CriteriaVal]#10
        noAgent = noAgentList[CriteriaVal]
        noObs = noObsList[CriteriaVal]
        neighborWeights = neighborWeightsList[CriteriaVal]
        playMode = playModeList

        ## initialize varaibles
        qtableList = []
        aPosList = []
        stateList = []
        rewardList = []
        doneList = []
        actionList = []
        nextStateList = []
        rewards_all_episodes = []
        visitCount = []

        ## Check if no of elements greater than the state space or not
        if (noAgent+noTarget+noObs+noFreeway)>= (gridHeight * gridWidth):
            print("Total number of elements (agents, targets, obstacles) exceeds grid position")
        else:
            # building environment
            env = Env(gridHeight, gridWidth, playMode, noTarget, noAgent, noObs, noFreeway)
            print('-------Initial Environment---------\n')
            env.render()
            print("\n")

        ## for each agent, initializing a Q-table with random Q-values
        for a in range(noAgent):
            qtableList.append(np.random.rand(env.stateCount, env.actionCount).tolist())

        ## hyperparameters
        totalEpisode = eList[CriteriaVal]
        gamma = 0.8 # discount factor
        epsilon = 0.08 #0.08 #exploration-exploitation
        intEpsilon = epsilon
        decay = 0.1 # decay of exploration-exploitation over episode
        stepsList = []
        alpha = 0.1 #learning rate

        ## initialize current experience harvesting budget (EHB) and current experience sharing budget (ESB)
        Behb = Behb_tot.copy()
        Besb = Besb_tot.copy()
        
        ## initialize visit count for each state
        for i in range(noAgent):
            visitCount.append([0 for x in range((gridWidth*gridHeight))])
        
          
            
            
        ## training loop
        for i in range(totalEpisode):
            print("epoch #", i+1, "/", totalEpisode)
            tPosList, aPosList, stateList, rewardList, doneList, oPosList, fPosList, courierNumber = env.reset(playMode, noTarget, noAgent, noObs,
                                                                       noFreeway, gridWidth, gridHeight, i, CriteriaVal,countVal,neighborWeights,totalEpisode,LoopVal)
            rewards_current_episode =[0 for a in range(noAgent)]
            doneList = [[a,'False'] for a in range(noAgent)]
            
            # render environment at the begining of every episode
            print("--------------Episode: ", i, " started----------------\n")
            if display=='on':
                env.render()
                print("\n")
            
            steps = 0
            completedAgent = []
            
            # uncomment only one line from below three lines according to your preference
            # while [0, 'True'] not in doneList: # ends when agent0 reaches goal
            while any('False' in sl for sl in doneList): # ends when all agents reach goal
#             while not any('True' in sl for sl in doneList): # ends when any agent reaches goal

                actionList = []
                if steps>(gridWidth*100):
                    break # break out of the episode if number of steps is too large to reach the goal.
                else:
                    steps +=1
                
                ## find which agents have completed
                completedAgent = [i for i, x in enumerate(doneList) if x[1]=='True']

                ## update visit count for this state and every agent
                for a in range(noAgent):
                    visitCount[a][stateList[a]] += 1

                ## find out neighbors starts---------------------------------------------------
                neighborDict = env.neighbors(noAgent, aPosList, gridWidth, gridHeight, flag)  
                neighborPosList = []
                for a in range(noAgent):
                    neighborsPrint = []
                    indNeighbor = []
                    for player in neighborDict[a]:
                        if a != aPosList.index(player):
                            indNeighbor.append(aPosList.index(player))
                        uniqueIndNeighbor = [*set(indNeighbor)]
                    neighborPosList.append(uniqueIndNeighbor)
                    uniqueIndNeighbor = []

                ## find out neighbors ends---------------------------------------------------
                
                
                
                
                # Experience harvesting (EH) and Experience Giving (EG) phase
                for a in range(noAgent):
                    ## calculate Pehc (experience harvesting confidence) based on visit count and budget. 
                    # If visit count is too high (i.e., >100000) or too low (<100) for any episode, set experience harvesting confidence to low (i.e., will not seek for advice)
                    if ((visitCount[a][stateList[a]]< 100) or (visitCount[a][stateList[a]]> 100000)):
                        Pehc = 0
                    else:
                        Pehc = (1/np.sqrt(visitCount[a][stateList[a]])) * (np.sqrt(Behb[a]/Behb_tot[a]))
                    
                    if ((Pehc > 0) and (Pehc < 0.1)) :
                        QNeighbor  = []
                        Behb[a] = Behb[a]-1
                        if a not in completedAgent:
                            neighborsOldQ = 0
                            selfOldQ = qtableList[a][stateList[a]]
                            neighborsOldQList = []
                            if neighborPosList[a] !=[]:  #if not empty list
                                for n in neighborPosList[a]:
                                    ## calculate Pesc (experience sharing confidence) based on visit count and budget
                                    if (visitCount[n][stateList[a]]> visitCount[a][stateList[a]]):
                                        Pesc = (1-(1/np.sqrt(visitCount[n][stateList[a]]))) * (np.sqrt(Besb[n]/Besb_tot[n]))
                                    else:
                                        Pesc = 0
                                    if Pesc > 0:
                                        Besb[n] = Besb[n]-1
                                        neighborsOldQ = qtableList[n][stateList[a]]
                                        #### Attacking (if any attacker presents)
                                        if n in Attacker:
                                            oldQAttacker = neighborsOldQ.copy()
                                            random.shuffle(neighborsOldQ)
                                            if oldQAttacker != neighborsOldQ:
                                                neighborsOldQ = swap(neighborsOldQ)
                                                ops = (add, sub)
                                                op = random.choice(ops)
                                                if op == add:
                                                    neighborsOldQ = [i+(goalReward/2) for i in neighborsOldQ]
                                                else:
                                                    neighborsOldQ = [i-(goalReward/2) for i in neighborsOldQ]
                                            else:
                                                neighborsOldQ = swap(neighborsOldQ)
                                                ops = (add, sub)
                                                op = random.choice(ops)
                                                if op == add:
                                                    neighborsOldQ = [i+(goalReward/2) for i in neighborsOldQ]
                                                else:
                                                    neighborsOldQ = [i-(goalReward/2) for i in neighborsOldQ]
                                        else:
                                            neighborsOldQ = neighborsOldQ
                                        neighborsOldQList.append(neighborsOldQ)
                                    else:
                                        neighborsOldQ = []
                                        neighborsOldQList.append(neighborsOldQ)
                                
                                # combining neighbors expereince
                                if any(neighborsOldQList):
                                    for i in range(4): # here 4 stands for four different actions
                                        elem = [item[i] for item in neighborsOldQList if item!=[]]
                                        # selecting the most appropiate advice
                                        QNeighbor.append(np.mean(elem))
                                            
                                    # Direct expereince aggregation
                                    qtableList[a][stateList[a]] = QNeighbor
                                else:
                                    qtableList[a][stateList[a]] = selfOldQ
                          
                
                # 1. select best action
                if np.random.uniform() < epsilon:
                    for a in range(noAgent):
                        actionList.append(env.randomAction())
                else:
                    for a in range(noAgent):
                        actionList.append(qtableList[a][stateList[a]].index(max(qtableList[a][stateList[a]])))
                        
                soqList = []   
                for a in range(noAgent):
                    soq = copy.deepcopy(qtableList[a])
                    soqList.append(soq)
                
                # 2. take the action and observe next state & reward
                nextStateList, rewardList, doneList, oPosList, courierNumber = env.step(actionList, doneList, noTarget, noAgent, noObs, noFreeway,
                                                               actionReward, obsReward, freewayReward, emptycellReward,
                                                               hitwallReward, completedAgent, goalReward)

                # 3. Calculate self Q-value
                for a in range(noAgent):
                    if a not in completedAgent:
                        qtableList[a][stateList[a]][actionList[a]] = ((qtableList[a][stateList[a]][actionList[a]] * (1 - alpha)) + (alpha * (rewardList[a] + gamma * max(qtableList[a][nextStateList[a]]))))
                        rewards_current_episode[a] += rewardList[a]
                        stateList[a] = nextStateList[a]
                    else:
                        qtableList[a][stateList[a]][actionList[a]] = qtableList[a][stateList[a]][actionList[a]]
                        rewards_current_episode[a] += rewardList[a]
                        stateList[a] = nextStateList[a]

                snqList = []
                for a in range(noAgent):
                    snq = copy.deepcopy(qtableList[a])
                    snqList.append(snq)

                # calcuating \Delta Q for convergence analysis
                for a in range(noAgent):
                    for p in range(len(soq)):
                        for q in range(len(soq[p])):
                            diff = abs(soqList[a][p][q] - snqList[a][p][q])
                            diffAvg1.append(diff)
                        diffAvg2.append(sum(diffAvg1)/len(diffAvg1))
                        diffAvg1 = []
                    diffAvg3.append(sum(diffAvg2)/len(diffAvg2))
                    diffAvg2 = []
   
            diffAvg4.append(sum(diffAvg3)/len(diffAvg3))
            diffAvg3 = []
            
            epsilon -= decay*epsilon # decaying exploration-exploitation probability for future episodes
            
            stepsList.append(steps)
            rewards_all_episodes.append(rewards_current_episode)
            print("\nDone in", steps, "steps".format(steps))
            time.sleep(sleep)

        stepsListFinal.append(stepsList)
        stepsList = []
        rewards_all_episodesFinal.append(rewards_all_episodes)
        rewards_all_episodes = []
        qtableListFinal.append(qtableList)
        qtableList = []
        diffAvg5.append(diffAvg4)
        diffAvg4 = []
    
    
    end = time.time()
    total_time = end-start
    print("Total Time taken: ",total_time) 
    
    s = stepsListFinal
    with open("./SG/"+str(fileName)+"_AdhocTD_Step", "wb") as Sp:   #Pickling
        pickle.dump(s, Sp)

    r = rewards_all_episodesFinal
    with open("./Reward/"+str(fileName)+"_AdhocTD_Reward", "wb") as Rp:   #Pickling
        pickle.dump(r, Rp)

    q = qtableListFinal
    with open("./Qtable/"+str(fileName)+"_AdhocTD_Qtable", "wb") as Qp:   #Pickling
        pickle.dump(q, Qp)
        
    c = diffAvg5
    with open("./Convergence/"+str(fileName)+"_AdhocTD_convergence", "wb") as Cp:   #Pickling
        pickle.dump(c, Cp)
    

    t = total_time
    with open("./TG/"+str(fileName)+"_AdhocTD_Time", "wb") as Tp:   #Pickling
        pickle.dump(t, Tp)   


    with open("./OutputFile/AdhocTD.txt", "a") as myfile:
        myfile.write("FileName: "+str(fileName)+" : AdhocTD, Time taken: "+str(total_time)+"\n | gridWidth: "+str(gridWidth)+" | gridHeight: "+str(gridHeight)+
                    " | playMode: "+str(playMode)+" | noTarget: "+str(noTarget)+" | noAgent: "+str(noAgent)+
                    " | noObs: "+str(noObs)+" | noFreeway: "+str(noFreeway)+
                    " | neighborWeights: "+str(neighborWeights)+" | totalEpisode: "+str(totalEpisode)+" | gamma: "+str(gamma)+
                    " | epsilon: "+str(intEpsilon)+" | decay: "+str(decay)+" | alpha: "+str(alpha)+
                    " | obsReward: "+str(obsReward)+" | freewayReward: "+str(freewayReward)+" | emptycellReward: "+str(emptycellReward)+
                    " | hitwallReward: "+str(hitwallReward)+" | Attacker: "+str(Attacker)+"\n\n\n")   



