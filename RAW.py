import numpy as np
import os
import utils
from tqdm import tqdm
import torch

def Raw_train():
    os.makedirs(r'.\Raw_data', exist_ok=True)
    RAW = np.load(r'.\data_d\Raw.npy', allow_pickle=True)
    _DATA = []
    for signal in tqdm(RAW):
        result_energy = utils.sliding_window_energy(signal)
        SIGNAL = torch.from_numpy(signal)
        ENERGY = torch.from_numpy(result_energy)
        SIGNAL = torch.unsqueeze(SIGNAL, dim=1)
        ENERGY = torch.unsqueeze(ENERGY, dim=1)
        OUTPUT = torch.cat((SIGNAL,ENERGY), dim=1)
        OUTPUT = OUTPUT.cpu().detach().numpy()
        _DATA.append(OUTPUT)
    _DATA = np.array(_DATA)
    np.save(r'.\Raw_data\Raw.npy', _DATA)
    
def Raw_eval(machine, id):
    os.makedirs(f'./Raw_data/{machine}/{id}', exist_ok=True)
    RAW = np.load(f'./data_e/{machine}/{id}/Raw.npy',allow_pickle=True)
    RAW_E = np.load(f'./data_e/{machine}/{id}/Raw_e.npy',allow_pickle=True)
    _DATA = []
    _DATA_E = []
    for signal in tqdm(RAW):
        result_energy = utils.sliding_window_energy(signal)
        SIGNAL = torch.from_numpy(signal)
        ENERGY = torch.from_numpy(result_energy)
        SIGNAL = torch.unsqueeze(SIGNAL, dim=1)
        ENERGY = torch.unsqueeze(ENERGY, dim=1)
        OUTPUT = torch.cat((SIGNAL,ENERGY), dim=1)
        OUTPUT = OUTPUT.cpu().detach().numpy()
        _DATA.append(OUTPUT)
    _DATA = np.array(_DATA)
    np.save(f'./Raw_data/{machine}/{id}/Raw.npy', _DATA)
    for signal in tqdm(RAW_E):
        result_energy = utils.sliding_window_energy(signal)
        SIGNAL = torch.from_numpy(signal)
        ENERGY = torch.from_numpy(result_energy)
        SIGNAL = torch.unsqueeze(SIGNAL, dim=1)
        ENERGY = torch.unsqueeze(ENERGY, dim=1)
        OUTPUT = torch.cat((SIGNAL,ENERGY), dim=1)
        OUTPUT = OUTPUT.cpu().detach().numpy()
        _DATA_E.append(OUTPUT)
    _DATA_E = np.array(_DATA_E)
    np.save(f'./Raw_data/{machine}/{id}/Raw_e.npy', _DATA_E)
             
