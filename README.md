# BRNES: Byzantine Robust Neighbor Experience Sharing in Differentially Private Multiagent Reinforcement Learning

This is the codification used in the IROS 2023 paper proposing BRNES framework as means of accelerating learning in Multiagent Systems composed of advisee and advisor agents. You are free to use all or part of the codes here presented for any purpose, provided that the paper is properly cited and the original authors properly credited. All the files here shared come with no warranties.


This project was built on Python 3.8. All the experiments are executed in the Predator-Prey (PP) domain, we included the version we used in the **Main/PP_environment** folder (slightly different from the standard PP domain). For the graph generation code you will need to install Jupyter Notebook (http://jupyter.readthedocs.io/en/latest/install.html).

## Files
The folder **Main** contains our implementation of all algorithms and experiments

The folder **Main/PP_environment** contains the Predator-Prey environment (also called a Pursuit domain) we used for experiments

Finally, the folder **ProcessedFiles** contains already processed files for graph printing and data visualization

## How to use <br />
First, install python 3.8 from https://www.python.org/downloads/release/python-380/<br />
Then open up your command terminal/prompt to run the following commands sequentially<br />
1. python RandomInit.py G N O E L Nw Ap D S M
2. python BRNES.py G N O E L Nw Ap D S M
3. python DARL.py G N O E L Nw Ap D S M
4. python AdhocTD.py G N O E L Nw Ap D S M
5. python Inference.py G N O E L Nw Ap D S M P DP Gap

where, <br />
G: Grid Height and Width (N x N)<br />
N: number of agents<br />
O: number of obstacles<br />
E: Total Episode<br />
L: number of times the code will run as a loop<br />
Nw: Neighbor weights [0,1]<br />
Ap: Attack Percentage [0,100]<br />
D: Display environment [on, off]<br />
S: Sleep (sec)<br />
M: Play mode [random, static]<br />
DP: LDP status [on, off] <br />
P: Privacy Budget (epsilon value) <br />
Gap: Inference Success Rate counting Gap <br />

<br />
Example:<br />

python RandomInit.py 15 10 3 2000 10 0.90 20 on 2 random<br />
python BRNES.py 15 10 3 2000 10 0.90 20 on 2 random<br />
python DARL.py 15 10 3 2000 10 0.90 20 on 2 random<br />
python AdhocTD.py 15 10 3 2000 10 0.90 20 on 2 random<br />
python Inference.py 15 10 3 2000 10 0.90 20 on 2 random on 1.0 10 <br />

<br /><br />
         
However, it might take a long time until all the experiments are completed. 
It may be of interest to run more than one algorithm at the same time if you have enough computing power. 
Also, note that, for each framework, if the agents do not attain goal within (GridSize*100) steps in a particular episode, the episode and environment will be reset to the next. <br /><br />

The **file name** associated with any experiment is appended into a log file (BRNES.txt) that resides inside "Main/OutputFile" directory.
The results (Steps to goal (SG), Time to goal (TG), Rewards, Convergence, Qtable) of any experiment are stored categorically by file name in "Main/SG", "Main/TG", "Main/Reward", "Main/Convergence", "Main/Qtable" respectively as a pickle file.
<br />
**Graph Generation and Reproduction**
1-a. Open processing.py file from "Main/" folder. Edit line 14-26 according to your experiments. Then run following command
	_python processing.py episode_num sg_gap reward_gap_
	where <br />
		episode_num = number of episode<br />
		sg_gap = plotting gap between SG values<br />
		reward_gap = plotting gap between Reward values<br />
Example: _python processing.py 1000 100 100_ <br /><br />

1-b. For Inference processing, run following command: _python Inference_processing.py_<br />
	Example: _python Inference_processing.py_ <br />
Your processed output will be stored inside the "Main/ProcessedOutput" folder in .csv format. Example output files are: ProcessedSG.csv, ProcessedReward.csv, ProcessedTG.csv, ProcessedConvergence.csv, ProcessedInference.csv<br /><br />
2. Then one-by-one run "Main/graphGenerator_SG.py", "Main/graphGenerator_Reward.py", "Main/graphGenerator_CostAndConvergence.py" through below example steps.<br /><br />
	a. Open Main/graphGenerator_SG.py and edit line 50-58 as per your experiment and graph generation preferences<br />
	b. run _python graphGenerator_SG.py episode_num gap_   (example: _python graphGenerator_SG.py 1000 100_)<br /><br />
	c. Open Main/graphGenerator_Reward.py and edit line 52-60 as per your experiment and graph generation preferences<br />
	d. run _python graphGenerator_Reward.py episode_num gap_  (example: _python graphGenerator_Reward.py 1000 100_)<br /><br />
	e. Open Main/graphGenerator_Convergence.py and edit line 50-55 as per your experiment and graph generation preferences<br />
	f. run _python graphGenerator_Convergence.py episode_num gap_   (example: _python graphGenerator_Convergence.py 5000 5_)<br /><br />
	g. Open Main/graphGenerator_TG.py and edit line 33-48 as per your experiment and graph generation preferences<br />
	h. run _python graphGenerator_TG.py_   (example: _python graphGenerator_TG.py_)<br /><br />
	i. Open Main/graphGenerator_Inference.py and edit line 29-33 as per your experiment and graph generation preferences<br />
	j. run _python graphGenerator_Inference.py_   (example: _python graphGenerator_Inference.py_)<br /><br />
	
Your output graphs will be stored in "Main/fig4(a-d)_SG.svg", "Main/fig4(a-d)_Reward.svg", "Main/fig5(b)part_Convergence.svg", 
"Main/fig6(a)_TG.svg", "Main/fig6b_Inference" <br /><br />
3. fig4e and fig5a in the paper are for SG, Reward, and Convergence for various values of privacy budget (epsilon). This can be generated easily by performing multiple experiments with desire values of epsilon.<br /><br />
4. For convenience, we include a "ProcessedFiles" folder that is already populated by the results of our experiments. <br />
	Processed outputs are already in the "ProcessedFiles/ProcssedOutput" folder.<br /><br />
	Simply, run the following commands from **./ProcessedFiles folder** to see the graphs we have included in our paper<br/>
	
	python graphGenerator_SG.py 1000 100
	python graphGenerator_Reward.py 1000 100
	python graphGenerator_Convergence.py 5000 5
	python graphGenerator_TG.py
	python graphGenerator_Inference.py 
	
	
	


## Contact
For questions about the codification or paper, please send an email to mdtamjidh@nevada.unr.edu or aralab2018@gmail.com.
