# %%
import csv
import numpy as np
import torch

wine_path = "../Ch04/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
wineq_numpy
# %%

col_list = next(csv.reader(open(wine_path), delimiter=';'))
wineq_numpy.shape, col_list
# %%

wineq = torch.from_numpy(wineq_numpy)
wineq.shape, wineq.dtype