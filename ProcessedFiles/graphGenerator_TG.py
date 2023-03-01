import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import seaborn as sns
import matplotlib.transforms as mtransforms

plt.rcParams['grid.alpha'] = 1
plt.rcParams["font.size"] = 15
plt.rcParams['grid.color'] = "#949292"
plt.rcParams['grid.linestyle'] = "--"
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
legend_properties = {'weight':'bold'}
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 1
plt.rcParams["legend.fontsize"] = 15

## Make sure path exists for this one line of code #################################
df = pd.read_csv("./ProcessedOutput/ProcessedTG.csv", delimiter=',')
################################################################################

#### Edit these lines according to your experiments and Graph generation preferences #################
AttackPercentage=['0','10', '30', '40']
y1 = []
y2 = []
y3 = []
for i in range(len(AttackPercentage)):
    y1.append(df["AdhocTD"+str(AttackPercentage[i])].values[0])
    y2.append(df["DARL"+str(AttackPercentage[i])].values[0])
    y3.append(df["BRNES"+str(AttackPercentage[i])].values[0])

df1 = pd.DataFrame({'AdhocTD':y1, 'DARL':y2, 'BRNES':y3})
df1 = df1.round(decimals = 1)
m1 = df1["AdhocTD"]
m2 = df1["DARL"]
m3 = df1["BRNES"]
fig, ax = plt.subplots(1,1, figsize=(6,4.5))
# yLim = 50000
################################################################################

x=AttackPercentage
index = np.arange(len(x))


ax.patch.set_facecolor('#ebebeb')
ax.patch.set_alpha(0.1)
bar_width = 0.25
opacity = 0.8
error_config = {'ecolor': '0.9'}

r1 = ax.bar(index-bar_width, m1, bar_width,
                 alpha=opacity,
                 color='c',
                 error_kw=error_config, label='AdhocTD')
r2 = ax.bar(index, m2, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config, label='DA-RL')
r3 = ax.bar(index + bar_width, m3, bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config, label='BRNES')
ax.xaxis.get_major_locator().set_params(nbins=5)
ax.yaxis.get_major_locator().set_params(nbins=20)
ax.set_xticks(index)
ax.set_xticklabels(x)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), fontsize=15,
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(6, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)

autolabel(r1)
autolabel(r2)
autolabel(r3)
bars = ax.patches
patterns =('\O','\O','\O','\O','x.','x.','x.','x.','.o','.o','.o','.o')
hatches = [p for p in patterns for i in range(len(df))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax.set_yscale('log')

ax.legend(loc='upper left', fancybox=True, shadow=True, framealpha = 0.9,
          borderpad=0.2,labelspacing=0.2, handlelength=1.5,columnspacing=0.5,handletextpad=0.2)
# ax.set_ylim(0,yLim) 
ax.set_xlabel('Percentage of the attackers (%)')
ax.set_ylabel('Time to goal (TG) in sec')

plt.tight_layout()


## Make sure path exists for this one line of code #################################
plt.savefig('./fig6(a)_TG.svg',bbox_inches='tight') 
################################################################################
plt.show()

