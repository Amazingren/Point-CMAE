
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.switch_backend('agg')
from matplotlib import rcParams

plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(1, 1, figsize=(10, 5))


ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Classification
epoch250 = [84.46, 84.80, 84.80, 84.42, 84.91, 84.66, 85.18, 85.29, 84.28]
epoch275 = [85.36, 84.59, 84.63, 84.28, 85.84, 85.95, 85.32, 85.22, 85.11]
epoch300 = [83.92, 84.66, 85.15, 84.03, 85.46, 85.46, 84.49, 85.11, 85.22]

epoch250_mean = [84.76, 84.76, 84.76, 84.76, 84.76, 84.76, 84.76, 84.76, 84.76]
epoch275_mean = [85.14, 85.14, 85.14, 85.14, 85.14, 85.14, 85.14, 85.14, 85.14]
epoch300_mean = [84.83, 84.83, 84.83, 84.83, 84.83, 84.83, 84.83, 84.83, 84.83]

plt.plot(ratio, epoch250, '-*', color='y', linewidth=2, alpha=0.7, label=r"Epoch250")
plt.plot(ratio, epoch275, '-^', color='r', linewidth=2, alpha=0.7, label=r"Epoch275")
plt.plot(ratio, epoch300, '-v', color='c', linewidth=2, alpha=0.7, label=r"Epoch300")

plt.plot(ratio, epoch250_mean, linestyle='--', marker='*', color='y', linewidth=2, alpha=0.7)
plt.plot(ratio, epoch275_mean, linestyle='--', marker='^', color='r', linewidth=2, alpha=0.7)
plt.plot(ratio, epoch300_mean, linestyle='--', marker='v', color='c', linewidth=2, alpha=0.7)


"""
# --------- DeiT ----------- #
#vit_s = [71.95, 61.96, 46.44, 29.94, 18.65]
#deit_s = [78.83, 73.47, 63.95, 49.94, 36.00]

deit_s_mjp   = [80.16, 78.68, 75.71, 70.61, 62.91]
deit_s_mjp_3 = [80.20, 79.29, 78.00, 75.71, 72.91]
deit_s_mjp_5 = [80.23, 79.80, 78.78, 77.38, 75.45]
deit_s_mjp_7 = [80.16, 79.81, 79.24, 78.45, 76.80]
deit_s_mjp_9 = [80.08, 79.92, 79.44, 78.70, 77.52]

#plt.plot(ratio, vit_s, '-*', linewidth=2, alpha=0.8, label="ViT-S")
#plt.plot(ratio, deit_s, '-^', linewidth=2, alpha=0.8, label="DeiT-S")
plt.plot(ratio, deit_s_mjp, '-s', color='g', linewidth=2, alpha=0.8, label=r"DeiT-S+MJP ($\gamma=0.1$)")
plt.plot(ratio, deit_s_mjp_3, '-P', color='b', linewidth=2, alpha=0.8, label=r"DeiT-S+MJP ($\gamma=0.3$)")
plt.plot(ratio, deit_s_mjp_5, '-*', color='y', linewidth=2, alpha=0.8, label=r"DeiT-S+MJP ($\gamma=0.5$)")
plt.plot(ratio, deit_s_mjp_7, '-^', color='c', linewidth=2, alpha=0.8, label=r"DeiT-S+MJP ($\gamma=0.7$)")
plt.plot(ratio, deit_s_mjp_9, '-v', color='r', linewidth=2, alpha=0.8, label=r"DeiT-S+MJP ($\gamma=0.9$)")

"""

# --------- Swin ----------- #
# swin = [79.75 , 72.76, 58.04, 31.16, 3.98]
# swin_mjp = [80.92, 79.68, 77.39, 71.30, 46.37]
# plt.plot(ratio, swin, '-^', linewidth=2, alpha=0.8, label="Swin-T")
# plt.plot(ratio, swin_mjp, '-s', color='g', linewidth=2, alpha=0.8, label=r"Swin-T+MJP")



leg = plt.legend(loc='best')
ax.set_ylabel('Classification Acc.(%)')
ax.set_xlabel(r'Masking ratio $r$')
#ax.set_ylim([40, 81])
#plt.show()

# -----
pdf = PdfPages("masking-ratio.pdf")
plt.savefig(pdf, format='pdf', bbox_inches='tight')
pdf.close()

pdf=None