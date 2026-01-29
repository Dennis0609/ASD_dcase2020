import numpy as np
import utils
import torch
from tqdm import tqdm

def develop():
    MACHINE = ['fan','pump','slider','valve']
    ID = ['id_00', 'id_02', 'id_04','id_06']
    for machine in MACHINE:
        for id in ID:
            lst = []
            Raw = np.load(f'./data_e/{machine}/{id}/Raw.npy',allow_pickle=True)
            for i in tqdm(Raw):
                i = torch.tensor(i)
                mel = utils.log_mel_energy(i)
                num = mel.shape[0]
                stand = np.zeros((num,224))
                for idx in range(0,num):
                    b = mel[idx:idx+1, :]
                    b = torch.squeeze(b,dim=0)
                    b = utils.gwap(b)
                    stand[idx,:] = b
                lst.append(stand)
            lst = np.vstack(lst)
            np.save(f'./data_e/{machine}/{id}/gwap.npy', lst)
    
def eval():
    MACHINE = ['fan','pump','slider','valve']
    ID = ['id_00', 'id_02', 'id_04','id_06']
    for machine in MACHINE:
        for id in ID:
            lst = []
            Raw_e = np.load(f'./data_e/{machine}/{id}/Raw_e.npy', allow_pickle=True)
            for i in tqdm(Raw_e):
                i = torch.tensor(i)
                i = i.to(torch.float32)
                mel = utils.log_mel_energy(i)
                num = mel.shape[0]
                stand = np.zeros((num,224))
                for idx in range(0,num):
                    b = mel[idx:idx+1, :]
                    b = torch.squeeze(b,dim=0)
                    b = utils.gwap(b)
                    stand[idx,:] = b
                lst.append(stand)
            lst = np.vstack(lst)
            np.save(f'./data_e/{machine}/{id}/gwap_e.npy', lst)
