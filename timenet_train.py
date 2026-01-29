import config
from Timenet import Timenet
from utils import time_decoder, ArcMarginProduct
import numpy as np
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
################################################
def get_model(device):
    model = Timenet().to(device)
    decoder = time_decoder().to(device)
    return model, decoder

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
##############################################
def loss(device):
    arc_loss = ArcMarginProduct().to(device)
    return arc_loss
def get_optimizer(model, decoder, epoch):
    optimizer =optim.AdamW([
        {'params': model.parameters()},
        {'params': decoder.parameters()}
        ], 
        lr=0.005,
        weight_decay=0.0005
        )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    return optimizer, lr_scheduler
##############################################
def Train(device, model, decoder, optimizer, arc_loss, Raw, Target):
    train_loss = 0.
    for i, raw in enumerate(tqdm(Raw)):
        loss = 0.
        raw = torch.from_numpy(raw).to(device)
        raw = raw.float()
        label = Target[i]
        label = torch.from_numpy(label).to(device)
        optimizer.zero_grad()
        
        out_ = model(raw)
        out = decoder(out_)
        loss1 = arc_loss(out, label)
        loss = loss1
        
        loss.backward()
        optimizer.step()
        train_loss +=loss
    train_loss = train_loss/len(Raw)
    return train_loss