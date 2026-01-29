import numpy as np
import torch
from data import audiodir,DCASE
import torchvision.transforms as transforms
import utils
import os
from tqdm import tqdm
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def train_load(machine, input_id):
    _DATA = []
    _LABEL = []
    ID = ['id_00', 'id_02', 'id_04','id_06']
    lst = []      
    lbl = []
    dir, label = audiodir(machine, input_id)
    new_list = [item for item in ID if item != input_id] 
    for Id in new_list: 
        x, y = audiodir(machine, Id)
        lst.append(x)
        lbl.append(y)
    flatten_dir = [item for sublist in lst for item in sublist] 
    flatten_label = [item for sublist in lbl for item in sublist]
    for idx, i in enumerate(dir):
        a = np.random.randint(0,len(flatten_dir))
        data1 = i
        label1 = label[idx]
        data2 = flatten_dir[a]
        label2 = flatten_label[a]
        mix_data,mix_label = utils.mix_up(data1, data2, label1, label2)
        _DATA.append(mix_data)
        _LABEL.append(mix_label) 
    
    dataset_train = DCASE(_DATA, _LABEL)
    Train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, drop_last=False)
    return Train
    
def transform(input_audio_numpy, only_mel=False):
    if only_mel == False:
        trans_index = np.random.randint(0, 3)
        if trans_index ==0:
            output_audio_numpy = utils.time_shift(input_audio_numpy, state=False)
            return output_audio_numpy
        elif trans_index == 1:
            output_audio_numpy = utils.AWGN(input_audio_numpy, state=False)
            return output_audio_numpy
        elif trans_index == 2:
            output_audio_numpy = utils.fade(input_audio_numpy, state=False)
            return output_audio_numpy
    else:
        output_audio_numpy = utils.to_2_img(input_audio_numpy,state=False)
    return output_audio_numpy
     
def apply_transform(machine, input_id):
    os.makedirs(f'./data_d/{machine}/{input_id}', exist_ok=True)
    Train = train_load(machine, input_id)
    ANCHOR = []
    POSITIVE = []
    NEGATIVE = []
    for Data, _ in tqdm(Train):
        batch_size = Data.shape[0]
        Anchor = torch.zeros((batch_size, 2, 224, 313))
        Positive = torch.zeros((batch_size, 2, 224, 313))
        Negative = torch.zeros((batch_size, 2, 224, 313))
        for x in range(len(Data)):
            data = Data[x]
            anchor = transform(data, only_mel=True)
            positive = transform(data)
            fake = generate_fake(anchor, machine, input_id)
            Anchor[x,:,:,:] = anchor
            Positive[x,:,:,:] = positive
            Negative[x,:,:,:] = fake
        Anchor = transforms.Resize((224,224))(Anchor)
        Positive = transforms.Resize((224,224))(Positive)
        Negative = transforms.Resize((224,224))(Negative)
        Anchor = Anchor.cpu().detach().numpy()
        Positive = Positive.cpu().detach().numpy()
        Negative = Negative.cpu().detach().numpy()
        ANCHOR.append(Anchor)
        POSITIVE.append(Positive)
        NEGATIVE.append(Negative)
    ANCHOR = np.array(ANCHOR)
    POSITIVE = np.array(POSITIVE)
    NEGATIVE = np.array(NEGATIVE)
    np.save(f'./data_d/{machine}/{input_id}/Anchor.npy', ANCHOR)
    np.save(f'./data_d/{machine}/{input_id}/Positive.npy', POSITIVE)
    np.save(f'./data_d/{machine}/{input_id}/Negative.npy', NEGATIVE)

