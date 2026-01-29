import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchaudio
import glob
import re
import itertools
from utils import get_10
import random
from tqdm import tqdm
import torch
import utils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_random(data,label, num_elements):
    random_data = random.sample(data, num_elements) 
    random_label = random.sample(label, num_elements) 
    return random_data, random_label

def random_cut(input_audio):        
    w = input_audio.shape[0]
    length = 114500
    time = length / 2
    start = time
    end = w - time
    mid = np.random.randint(start, end+1)
    x1 = mid - time
    x2 = mid + time
    data = input_audio[int(x1):int(x2)]
    return data
    
def func(string):
    __pattern = r'normal_id_(\d+)_'
    match = re.match(__pattern, string)
    if match:
        return int(match.group(1))
    else:
        return -1 
    
def judge(key, idx):
    if key in ['fan', 'pump', 'slider', 'valve']:
        if idx==0:
            return 0
        elif idx==2:
            return 1
        elif idx==4:
            return 2
        elif idx==6:
            return 3
    else:
        if idx==1:
            return 0
        elif idx==2:
            return 1
        elif idx==3:
            return 2
        elif idx==4:
            return 3
        
def get_machine_id_list(machine, state, base_dir = r'C:\Users\admin\Desktop\liao\data',ext="wav"):
    traindir = base_dir + '\\' + machine + '\\' + 'train'
    test_dir = base_dir + '\\' + machine + '\\' + 'test'
    if state == 'train':
        dir_path = os.path.abspath("{dir}/*.{ext}".format(dir=traindir, ext=ext))
        file_paths = sorted(glob.glob(dir_path))
        machine_id_list = sorted(list(set(itertools.chain.from_iterable(
            [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    else:
        dir_path = os.path.abspath("{dir}/*.{ext}".format(dir=test_dir, ext=ext))
        file_paths = sorted(glob.glob(dir_path))
        machine_id_list = sorted(list(set(itertools.chain.from_iterable(
            [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list

def audiodir(machine, id, base_dir = r'C:\Users\admin\Desktop\liao\data'):
    traindir = base_dir + '\\' + machine + '\\' + 'train'  
    label = []
    data = []
    dictionary = {  
        'fan': '0',  
        'pump': '4',  
        'slider': '8',
        'valve': '12'
    } 
    num = dictionary[machine]
    list_dir = os.listdir(traindir)
    for dir in list_dir:
        if id in dir:
            dir_address = traindir + '\\' + dir   
            labels = np.zeros((16))                               
            x,sr = torchaudio.load(dir_address)
            x = x.mean(axis=0)
            x = get_10(x)
            data.append(x)
            _idx = judge(machine, int(func(dir)))
            _num = int(_idx) + int(num)
            labels[_num] = 1
            label.append(labels)
    return data,label

def file_list_generator(machine, id, state):
    base_dir =  r'C:\Users\admin\Desktop\liao\data'
    target_dir = base_dir + '//' + machine + '//' + state
    list_d =os.listdir(target_dir)
    if state == 'train':
        list_for_id = []
        label = []
        for dir in list_d:
            if id in dir:
                dir = target_dir + '/' + dir
                x,sr = torchaudio.load(dir)
                x = x.mean(axis=0)
                x = get_10(x)
                list_for_id.append(x)
                label.append(np.array([int(id[-1:])]))
    else:
        list_for_id = []
        label = []
        key = 'normal'
        for dir in list_d:
            if key in dir and id in dir:
                dir = target_dir + '/' + dir
                x,sr = torchaudio.load(dir)
                x = x.mean(axis=0)
                x = get_10(x)
                list_for_id.append(x)
                label.append(1)
            elif id in dir and key not in dir:
                dir = target_dir + '/' + dir
                x,sr = torchaudio.load(dir)
                x = x.mean(axis=0)
                x = get_10(x)
                list_for_id.append(x)
                label.append(0)
    return list_for_id, label

class DCASE(Dataset):                                    
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
def train_load():            
    MACHINE = ['fan','pump','slider','valve']
    _DATA = []
    _LABEL = []
    for machine in MACHINE:
        ID = ['id_00', 'id_02', 'id_04','id_06']
        for input_id in ID:
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
    Train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, drop_last=False)
    return Train

def id_load(machine,id):
    data_e,id_e = file_list_generator(machine, id, state='train')
    data_t,id_t = file_list_generator(machine, id, state='test')
    
    dset_e = DCASE(data_e,id_e)
    dset_t = DCASE(data_t,id_t)
    D = torch.utils.data.DataLoader(dset_e, batch_size=128, shuffle=True, drop_last=False)
    T = torch.utils.data.DataLoader(dset_t, batch_size=128, shuffle=True, drop_last=False)
    return D,T

def transform(input_audio_numpy, only_mel=False):
    if only_mel == False:
        trans_index = np.random.randint(0, 4)
        if trans_index ==0:
            output_audio_numpy = utils.time_shift(input_audio_numpy)
            return output_audio_numpy
        elif trans_index == 1:
            output_audio_numpy = utils.AWGN(input_audio_numpy)
            return output_audio_numpy
        elif trans_index == 2:
            output_audio_numpy = utils.fade(input_audio_numpy)
            return output_audio_numpy
        elif trans_index == 3:
            output_audio_numpy = utils.pitch_shift(input_audio_numpy)
            return output_audio_numpy
    else:
        output_audio_numpy = utils.to_2_img(input_audio_numpy,state=False)
    return output_audio_numpy

def apply_transforms():
    os.makedirs(r'.\data_d', exist_ok=True)
    Train = train_load()
    ANCHOR = []
    DATA_1 = []
    DATA_2 = []
    LABEL = []
    RAW = []
    for Data, Label in tqdm(Train):
        Label = Label.cpu().detach().numpy()
        LABEL.append(Label)
        _Data = Data.cpu().detach().numpy()
        RAW.append(_Data)
        batch_size = Data.size()[0]
        Anchor = torch.zeros((batch_size, 2, 224, 313))
        Data_1 = torch.zeros((batch_size, 2, 224, 224))
        Data_2 = torch.zeros((batch_size, 2, 224, 224))
        for x in range(len(Data)):
            data = Data[x]
            cut_data = random_cut(data)
            
            anchor = transform(data, only_mel=True)
            data_1 = transform(cut_data)
            data_2 = transform(cut_data)
            
            Anchor[x,:,:,:] = anchor
            Data_1[x,:,:,:] = data_1
            Data_2[x,:,:,:] = data_2
        Anchor = transforms.Resize((224,224))(Anchor)
        Anchor = Anchor.cpu().detach().numpy()
        Data_1 = Data_1.cpu().detach().numpy()
        Data_2 = Data_2.cpu().detach().numpy()
        ANCHOR.append(Anchor)
        DATA_1.append(Data_1)
        DATA_2.append(Data_2)
    ANCHOR = np.array(ANCHOR)
    DATA_1 = np.array(DATA_1)
    DATA_2 = np.array(DATA_2)
    LABEL = np.array(LABEL)
    RAW = np.array(RAW)
    np.save(r'.\data_d\Train1.npy', DATA_1)
    np.save(r'.\data_d\Train2.npy', DATA_2)
    np.save(r'.\data_d\Anchor.npy', ANCHOR)
    np.save(r'.\data_d\target.npy', LABEL)
    np.save(r'.\data_d\Raw.npy', RAW)
    
               
def list_to_vector(machine, id):
    os.makedirs(f'./data_e/{machine}/{id}', exist_ok=True)
    D,T = id_load(machine, id)
    DATA = []
    LABEL = []
    RAW = []
    for data, label in tqdm(D):
        _Data = data.cpu().detach().numpy()
        RAW.append(_Data)
        label = label.cpu().detach().numpy()
        LABEL.append(label)
        batch_size = data.size()[0]
        Data = torch.zeros((batch_size,2,224,313))
        for x in range(len(data)):
            Data[x,:,:,:]=transform(data[x],only_mel=True)
        Data = transforms.Resize((224,224))(Data)
        Data = Data.cpu().detach().numpy()
        DATA.append(Data)
    DATA = np.array(DATA)
    LABEL = np.array(LABEL)
    RAW=  np.array(RAW)
    DATA_E = []
    LABEL_E = []
    RAW_E = []
    for data,label in tqdm(T):
        _Data = data.cpu().detach().numpy()
        RAW_E.append(_Data)
        label = label.cpu().detach().numpy()
        LABEL_E.append(label)
        batch_size = data.size()[0]
        Data = torch.zeros((batch_size,2,224,313))
        for i in range(len(data)):
            Data[i,:,:,:]=transform(data[i],only_mel=True)
        Data = transforms.Resize((224,224))(Data)
        Data = Data.cpu().detach().numpy()
        DATA_E.append(Data)
    DATA_E = np.array(DATA_E)    
    LABEL_E = np.array(LABEL_E)  
    RAW_E = np.array(RAW_E)              
    np.save(f'./data_e/{machine}/{id}/data.npy',DATA)
    np.save(f'./data_e/{machine}/{id}/label.npy',LABEL)
    np.save(f'./data_e/{machine}/{id}/data_e.npy',DATA_E)
    np.save(f'./data_e/{machine}/{id}/label_e.npy',LABEL_E)
    np.save(f'./data_e/{machine}/{id}/Raw.npy',RAW)
    np.save(f'./data_e/{machine}/{id}/Raw_e.npy',RAW_E)
