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

fig = plt.figure(constrained_layout=False,figsize=(25,15))
gs= GridSpec(3, 4, figure=fig, wspace=0.1, hspace = 0.3)
ax0 = plt.subplot(gs.new_subplotspec((2, 0)))
ax1 = plt.subplot(gs.new_subplotspec((2, 1)),sharey=ax0)
ax2 = plt.subplot(gs.new_subplotspec((2, 2)),sharey=ax0)
ax3 = plt.subplot(gs.new_subplotspec((2, 3)),sharey=ax0)
plt.setp(ax1.get_yticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
axList = [[ax0], [ax1], [ax2], [ax3]]
ax0.set_xlabel('Episodes')
ax1.set_xlabel('Episodes')
ax2.set_xlabel('Episodes')
ax3.set_xlabel('Episodes')
ax0.set_ylabel('Convergence')
labelList = ['(a)', '(b)', '(c)', '(d)']

episodeNum = int(sys.argv[1])
convGap = int(sys.argv[2])


x3 = list(np.linspace(10,episodeNum+10,int(episodeNum/convGap), endpoint=False))

## Make sure path exists for this one line of code #################################
df = pd.read_csv("./ProcessedOutput/ProcessedConvergence.csv", delimiter=',')
################################################################################

#### Edit these lines according to your experiments and Graph generation preferences #################
frameworkName = ['BRNES','AdhocTD','DARL']
AttackPercentage=['0','10', '30', '40']
attackerNo = ['No Attacker', '10% Attacker', '30% Attacker', '40% Attacker']
category = ['Step', 'Reward', 'convergence']
lineStyle = ['s-','v-','o-','^-.','*--','s-']
color = ['red','cyan','black','magenta','blue','green']
markerFace = ['none','cyan','none','magenta','blue','green']
markerEdge = ["black","black","black","black","black","black"]
legend = ['BRNES','AdhocTD', 'DA-RL']
################################################################################

### Convergence #####
for i in range(len(AttackPercentage)):
    for j in range(len(frameworkName)):
        axList[i][0].plot(x3, df[str(frameworkName[j])+str(AttackPercentage[i])], lineStyle[j], color = color[j], markerfacecolor=markerFace[j],
		                              markeredgecolor=markerEdge[j], label = legend[j])
        axList[i][0].set_title(str(attackerNo[i]), loc = "right")
        axList[i][0].legend(loc ='center', bbox_to_anchor=(0.8, 0.5), #0.25,-1,
		          ncol=1, fancybox=True, shadow=True,
		           borderpad=0.2,labelspacing=0.2, handlelength=0.9,columnspacing=0.8,handletextpad=0.3)
        axList[i][0].tick_params(axis='x')
        axList[i][0].tick_params(axis='y')
        trans = mtransforms.ScaledTranslation(-1/72, 1/72, fig.dpi_scale_trans)
        axList[i][0].text(0.5, -0.35, labelList[i], transform=axList[i][0].transAxes + trans)
        




## Make sure path exists for this one line of code #################################
plt.savefig('./fig5(b)part_Convergence.svg',bbox_inches='tight') 
################################################################################

plt.show()




