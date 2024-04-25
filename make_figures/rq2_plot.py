"""
Make plots for RQ2: effect of timesteps

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['savefig.format'] = 'pdf'
plt.rc('font', family='serif')
fs = 26
#
# File paths
#
file_ddh = '../results/rq2/metrics_ddh.csv'
file_lsb = '../results/rq2/metrics_lsb.csv'
file_udh = '../results/rq2/metrics_udh.csv'
#
# Reading the files
#
df_ddh = pd.read_csv(file_ddh)
df_lsb = pd.read_csv(file_lsb)
df_udh = pd.read_csv(file_udh)

plt.figure(figsize=(10, 7))
#
# Defining colors for each dataset
#
colors = {'DDH': ('blue', 'blue'), 'LSB': ('green', 'green'), 'UDH': ('darkorange', 'darkorange')}
texture = {'DDH': "^", 'LSB': ".", 'UDH': "x"}
#
# Plotting for each dataset with specified colors
#
for df, label in zip([df_ddh, df_udh, df_lsb], ['DDH', 'UDH', 'LSB']):  # Reordered to match color specification
    ip_color, se_color = colors[label]
    dot = texture[label]
    plt.plot([i for i in range(25, 1001, 25)], df['ncc_chat'], label=f'{label} IP', marker=dot, color=ip_color)
    plt.plot([i for i in range(25, 1001, 25)], df['ncc_secret'], "--", label=f'{label} SE', marker=dot, color=se_color)

plt.axhspan(0.30, 0.95, color='red', alpha=0.18)
# Adding titles and labels
plt.xlabel('Timesteps (t)', fontsize=fs)
plt.ylabel('NCC', fontsize=fs)
plt.legend(loc="upper left", ncol=3, fontsize=fs-6,  bbox_to_anchor=(0.02, 1.2))
plt.tick_params(axis='x', labelsize=fs)
plt.tick_params(axis='y', labelsize=fs)
#
# Save
#
plt.grid(True)
plt.savefig("../results/rq2_plot_ncc.pdf", bbox_inches='tight', dpi=500)