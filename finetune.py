from mobilenetv2 import Mobilenetv2
from utils import Decoder as Decoder
from utils import TripletLoss
import os
import torch.optim as optim
import torch.nn as nn
import config
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from tqdm import tqdm

################################################
def get_model(device):
    model = Mobilenetv2().to(device)
    decoder = Decoder().to(device)
    return model, decoder

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
##############################
def get_optimizer(model, decoder, epoch):
    optimizer = optim.AdamW(decoder.parameters(),
        lr=0.001,
        weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    return optimizer, lr_scheduler
def get_loss(device):
    triplet_loss = TripletLoss().to(device)
    cos_loss = nn.CosineSimilarity(dim=1)
    return triplet_loss,cos_loss
###############################################################
def finetune(device, model, decoder, triplet_loss, cos_loss, optimizer, Anchor, Positive, Negative):
    model = model.to(device)
    decoder = decoder.to(device)
    finetune_loss = 0.
    for i, anchor in enumerate(tqdm(Anchor)):
        neg = Negative[i]
        pos = Positive[i]
        anchor = torch.from_numpy(anchor).to(device)
        neg = torch.from_numpy(neg).to(device)
        pos = torch.from_numpy(pos).to(device)
        
        out = model(anchor)
        out1 = model(neg)
        out2 = model(pos)
        out = decoder(out)
        out1 = decoder(out1)
        out2 = decoder(out2)
        loss1 = triplet_loss(out,out2,out1)
        loss2 = (-cos_loss(out,out2).mean() + cos_loss(out, out1).mean()) / 2
        loss = (0.01 + loss1)* torch.exp(loss2 + 0.005)
        
        finetune_loss +=loss.item()
        loss.backward()
        optimizer.step()
    finetune_loss = finetune_loss/len(Anchor)
    return finetune_loss

