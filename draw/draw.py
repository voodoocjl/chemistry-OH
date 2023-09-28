import pennylane as qml
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from Arguments import Arguments
import warnings
import matplotlib.cbook
from ChemModel import translator, quantum_net
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

dir_path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(dir_path)
# dataset_file = os.path.join(dir_path, 'mosi_dataset')

args = Arguments()

# lst,res, mae=[], [], []

# df = pd.read_excel('result.xlsx', sheet_name='Sheet1')
# for index, row in df.iterrows():
#     lst.append(eval(row['arch_code']))
#     mae.append(row['test_mae'])

# ChooseNum=len(lst)

# code = torch.tensor(lst).numpy()

# for i in range(ChooseNum):
#     design = translator(lst[i])
#     weight = nn.Parameter(torch.rand(design['layer_repe'] * args.n_qubits * 2))
#     fig, _ = qml.draw_mpl(quantum_net, decimals=1, style="black_white", fontsize="x-small")(weight, design = design)
#     fig.suptitle(str(code[i]) + ': ' + str(mae[i]), fontsize="xx-large")   
#     plt.savefig('results_'+str(i)+'.jpg')    
#     # fig.show()
#     # plt.clf()

# net = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 3, 7, 6, 3, 1, 1, 10, 6, 5, 6, 6, 7]
# net = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 3, 7, 6, 3, 1, 1, 6, 7, 5, 6, 6, 7]
# net = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 3, 7, 6, 3, 1, 2, 6, 7, 5, 6, 6, 7]
# net = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 'n', 7, 6, 3, 1, 2, 6, 7, 5, 6, 'n', 11]
net = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 3, 'n', 6, 'n', 1, 2, 'n', 7, 5, 6, 'n', 'n']
design = translator(net)
weight = nn.Parameter(torch.rand(design['layer_repe'] * args.n_qubits * 2))
fig, _ = qml.draw_mpl(quantum_net, decimals=1, style="black_white", fontsize="x-small")(weight, design = design)
plt.savefig('draw/results.jpg')