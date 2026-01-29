import utils
from mobilenetv2 import Mobilenetv2
from utils import contrastive_block as contrastive_block
from utils import Decoder as Decoder
import torch
import torch.optim as optim
import numpy as np
import config
import random
import os
import psutil
import pynvml
from tqdm import tqdm
import matplotlib.pyplot as plt
################################################
def get_model(device):
    model = Mobilenetv2().to(device)
    decoder = Decoder().to(device)
    block = contrastive_block().to(device)
    return model, decoder, block

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
################### get info #####################
def get_gpu_mem_info(gpu_id):
    pynvml.nvmlInit()
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free
def get_cpu_mem_info():
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(
        os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_process_used, mem_free
def text_create(msg):
    os.makedirs(r'.\debug', exist_ok=True)
    desktop_path = r".\debug"
    full_path = desktop_path + '\\' + 'debug.txt'
    file = open(full_path, 'a')
    file.write(msg)
    file.write("\n")
    file.close()
##############################################
def loss(device):
    arc_loss = utils.ArcMarginProduct().to(device)
    return arc_loss
def get_optimizer(model, decoder, block, epoch):
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': block.parameters()}
        ],
        lr = 0.05,
        momentum=0.9,
        weight_decay=0.005
        )
    dowm_optimizer =optim.AdamW([
        {'params': model.parameters()},
        {'params': decoder.parameters()}
        ], 
        lr=0.05,
        weight_decay=0.0005
        )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    down_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dowm_optimizer, T_max=epoch)
    return optimizer, lr_scheduler, dowm_optimizer, down_lr_scheduler
################################################
def pre_Train(device, model, block, optimizer, Train1, Train2):
    model = model.to(device)
    block = block.to(device)
    train_loss = 0.
    for i, data1 in enumerate(tqdm(Train1)):
        loss = 0.
        optimizer.zero_grad()
        data2 = Train2[i]
        data1 = torch.from_numpy(data1).to(device)
        data2 = torch.from_numpy(data2).to(device)
        
        out1 = model(data1)
        out2 = model(data2)
        d1, d2 = block(out1, out2)
        loss1 = d1 + d2
        
        loss = loss1 
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss = train_loss/len(Train1)
    return train_loss

def adapt_Train(device, model, decoder, optimizer, arc_loss, Anchor, Label):
    model = model.to(device)
    decoder = decoder.to(device)
    adaptrain_loss = 0.
    for i, anchor in enumerate(tqdm(Anchor)):
        loss = 0.
        target = Label[i]
        optimizer.zero_grad()
        anchor = torch.from_numpy(anchor).to(device)
        target = torch.from_numpy(target).to(device)
        
        out1_ = model(anchor)
        out1 = decoder(out1_)
        loss1 = arc_loss(out1,target)
        loss = 0.1* loss1
        
        adaptrain_loss +=loss
        loss.backward()
        optimizer.step()
    adaptrain_loss = adaptrain_loss/len(Anchor)
    return adaptrain_loss
    