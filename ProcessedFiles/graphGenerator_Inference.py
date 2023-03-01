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
plt.rcParams["lines.markersize"] = 8
plt.rcParams["legend.fontsize"] = 20
plt.rcParams['grid.color'] = "#949292"
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
legend_properties = {'weight':'bold'}

plt.rcParams["lines.markersize"] = 8

#### Edit these lines according to your experiments and Graph generation preferences #################
lineStyle = ['s-','v-','o-','^-.','*--','s-']
color = ['red','cyan','black','magenta','blue','green']
markerFace = ['none','cyan','none','magenta','blue','green']
markerEdge = ["black","black","black","black","black","black"]
legend = ['BRNES','AdhocTD', 'DA-RL']
######################################################################################################

fig, ax = plt.subplots(1,1, figsize=(6,4.5))


df1 = pd.read_csv("./ProcessedOutput/ProcessedInference.csv", delimiter=',')

for i in range(len(df1.columns)):
    ax.plot(df1[df1.columns[i]], lineStyle[i], color=color[i], label=df1.columns[i], markerfacecolor ='none', markeredgecolor='black')
    
ax.set_xlabel('Episodes')
ax.set_ylabel('Attack Success Rate (%)')
ax.legend(loc ='upper left', bbox_to_anchor=(0, 1.02), #0.25,-1,
                      ncol=1, fancybox=True, shadow=True,
                       borderpad=0.1,labelspacing=0.1, handlelength=0.8,columnspacing=0.8,handletextpad=0.2)

plt.savefig('./fig6b_Inference.svg',bbox_inches='tight') 
plt.show()


