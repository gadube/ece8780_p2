import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()
sns.set_style("whitegrid", {'axes.grid' : True})
sns.set_color_codes()
#sns.set_style("white")
sns.set_context("notebook", font_scale=2.0, rc={"lines.linewidth": 2.5, 'lines.markeredgewidth': 1., 'lines.markersize': 10})
# axis labels
font = {'family' : 'serif'}
mpl.rc('font', **font)

df = pd.read_csv("profiling.csv")
#df = df[df["Input Image"] != 'lemons_3456_2304.png']
df = df[df["Input Image"] != 'peppers_506_326.jpeg']
#s_df = df[df['Version'] != "CUDA"]
#cu_df = df[df['Version'] != "serial"]
s_df = df[df['Name'] == 'serialRecombineChannels']
cu_df = df[df['Name'] == 'recombineChannels']
kernel = cu_df[cu_df['Name'] != 'CUDA memcpy HtoD']
kernel = kernel[kernel['Name'] != 'CUDA memcpy DtoH']
dtoh = df[df['Name'] == '[CUDA memcpy DtoH]']
htod = df[df['Name'] == '[CUDA memcpy HtoD]']


transfers = pd.concat([dtoh,htod])
#print(dtoh)

transfers['Duration'] = transfers['Duration'].astype(float)
transfers['Duration'] = transfers['Duration'] * 10**6
kernel['Duration'] = kernel['Duration'].astype(float)
kernel['Duration'] = kernel['Duration'] * 10**6

bar_width = .25
colors = ['c', 'y', 'b', 'r', 'g', 'm']
opacity = 1
blocks = [4, 8, 16, 32]
images = s_df['Input Image'].unique()
gpus = s_df['Device Name'].unique()

for gpu in gpus:
    for image in images:
        avg_dtoh = dtoh.loc[(dtoh['Device Name'] == gpu) & (dtoh["Input Image"] == image)]['Duration'].mean()
        print(gpu, image,"Average Device to Host Memcpy (us): ", avg_dtoh * 10**6)
        avg_htod = htod.loc[(htod['Device Name'] == gpu) & (htod["Input Image"] == image)]['Duration'].mean()
        print(gpu, image ,"Average Host to Device Memcpy (us): ", avg_htod * 10**6)



for gpu in ['Tesla V100-PCIE']:
    plt.clf()
    xticks = []
    for iw, blksz in enumerate(blocks):
        ucBar_pos=iw * (1 + 2*len(images) + 1.5) * bar_width

        for ir, img in enumerate(images):
            serial_time = s_df.loc[(s_df['Block Size'] == blksz) & (s_df['Input Image'] == img) & (s_df['Device Name'] == gpu)]['Duration'].mean()
            xpos = ucBar_pos + (ir + 1) * bar_width
            plt.bar(xpos, serial_time, bar_width, color=colors[ir])

        for ir, img in enumerate(images):
            cu_time = cu_df.loc[(cu_df['Block Size'] == blksz) & (cu_df['Input Image'] == img) & (cu_df['Device Name'] == gpu)]['Duration'].mean()
            xpos = ucBar_pos + (len(images) + ir + 1) * bar_width
            plt.bar(xpos, cu_time, bar_width, color=colors[ir], hatch='//')

        xticks.append((xpos + ucBar_pos) / 2)

    plt.xticks(xticks, blocks)
    x=plt.bar(0,0,0,color='w',edgecolor='k',hatch='//', label='CUDA')
    legend = [mpl.patches.Patch(facecolor='w',edgecolor='k', label='Serial'), x]
    images = ['Baboon','Dog','Lemons','Lena']
    legend.extend( [mpl.patches.Patch(color=colors[i], label=r) for i, r in enumerate(images)])
    plt.legend(handles=legend, bbox_to_anchor=(.5,1.10), loc='upper center', ncol=6)
    plt.semilogy(base=10)
    plt.ylim(10**-6, 10**-1)
    plt.ylabel('Duration (s) - Lower is Better', weight="bold")
    plt.xlabel('Block Size', weight="bold")
    #plt.show()
