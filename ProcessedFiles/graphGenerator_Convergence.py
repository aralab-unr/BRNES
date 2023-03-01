#####----------------Disclaimer----------------------------------------------###########
##### This convergence graph plotting script is a little bit different from  ###########
##### Main/grapGeneration_Convergence.py since here, we only focus on        ###########
##### plotting Convergence for all frameworks (BRNES, AdhocTD, DA-RL) in one ###########
##### single environment (i.e., medium-scale environment) with 30% attacker  ###########
#####------------------------------------------------------------------------###########


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import seaborn as sns
import matplotlib.transforms as mtransforms

sns.set_theme(context = "notebook",style="ticks", palette='bright')
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.autolayout"] = False
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 0
plt.rcParams["legend.fontsize"] = 20
plt.rcParams['grid.color'] = "#949292"
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
legend_properties = {'weight':'bold'}


episodeNum = int(sys.argv[1])
convGap = int(sys.argv[2])



x3 = list(np.linspace(10,episodeNum+10,int(episodeNum/convGap), endpoint=False))

## Make sure path exists for this one line of code #################################
df = pd.read_csv("./ProcessedOutput/ProcessedConvergence.csv", delimiter=',')
################################################################################

fig, ax = plt.subplots(1,1, figsize=(6,4.5))

#### Edit these lines according to your experiments and Graph generation preferences #################
frameworkName = ['BRNES','AdhocTD','DARL']
lineStyle = ['s-','v-','o-','^-.','*--','s-']
color = ['red','cyan','black','magenta','blue','green']
markerFace = ['none','cyan','none','magenta','blue','green']
markerEdge = ["black","black","black","black","black","black"]
legend = ['BRNES','AdhocTD', 'DA-RL']
######################################################################################################

for j in range(len(frameworkName)):
    ax.plot(x3, df[str(frameworkName[j])], lineStyle[j], color = color[j], markerfacecolor=markerFace[j],
		                              markeredgecolor=markerEdge[j], label = legend[j])
    
ax.legend(loc ='center', bbox_to_anchor=(0.8, 0.5), #0.25,-1,
		          ncol=1, fancybox=True, shadow=True,
		           borderpad=0.2,labelspacing=0.2, handlelength=0.9,columnspacing=0.8,handletextpad=0.3)
ax.tick_params(axis='x')
ax.tick_params(axis='y')

 




## Make sure path exists for this one line of code #################################
plt.savefig('./fig5(b)part_Convergence.svg',bbox_inches='tight') 
################################################################################

plt.show()




