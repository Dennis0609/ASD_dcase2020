import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
############################  预处理  ######################################
def get_10(input_data_numpy, sample_rate=16000):
    start_sample = 0
    target_samples = int(sample_rate * 10)
    first_10_seconds = input_data_numpy[start_sample :target_samples]
    return first_10_seconds 
 
def gwap(data, decay=1, dim=1):
    data = np.sort(data, axis=dim)[:, ::-1]
    gwap_w = decay ** np.arange(data.shape[dim])
    sum_gwap_w = np.sum(gwap_w)
    data = data * gwap_w
    out = np.sum(data, axis=dim)
    out = out / sum_gwap_w
    out = torch.tensor(out)
    return out
#########################################
###############  RAW  #################
def sliding_window_energy(matrix, window_size=2):  
    rows, cols = matrix.shape  
    result_matrix = np.zeros((rows, cols))  
    step_size = 1  
    num_windows = (cols - window_size) // step_size + 1 
    for i in range(0, num_windows, step_size):  
        window = matrix[:, i:i + window_size]  
        energy = np.sum(np.abs(window)**2, axis=1)  
        result_matrix[:, i % window_size] = energy
    return result_matrix  
########################################################
##################################### AUG #################################
'''
注意################
state = True
目的为配合预处理阶段进行数据加窗
在下游使用时state设置为False
'''
def pitch_shift(input_data_numpy, sr=16000, state = True):
    audio_pitch = torchaudio.functional.pitch_shift(waveform=input_data_numpy, sample_rate=sr, n_steps=-20)
    audio_pitch_shift = to_2_img(audio_pitch, state)
    return audio_pitch_shift
def time_shift(input_data_numpy,shift_max = 56, sr=16000, state= True):
    shift_rate = np.random.uniform(0,shift_max)
    shift_rate = int(56*sr)
    y_shift = np.roll(input_data_numpy.detach().numpy(), shift_rate)
    audio_time_shift = to_2_img(torch.from_numpy(y_shift), state)
    return audio_time_shift
def AWGN(input_data_numpy,SNR_min=-10, SNR_max=10, sr=16000, state = True):
    SNR = np.random.uniform(SNR_min,SNR_max)
    y = input_data_numpy.detach().numpy()
    RMS = np.sqrt(np.mean(y**2))
    STD_n = np.sqrt(RMS**2/(10**(SNR/10)))
    noise = np.random.normal(0,STD_n,y.shape[0])
    y_noise = y + noise
    y_noise = torch.from_numpy(y_noise)
    S = to_2_img(y_noise.float(), state)
    return S
def fade(input_data_numpy, sr=16000, state = True):
    fade_shape_num = np.random.randint(0,2)
    if fade_shape_num == 0:
      shape = 'exponential'
    elif fade_shape_num == 1:
      shape = 'half_sine'
    fade_in = np.random.randint(0,input_data_numpy.shape[0]/2)
    fade_out = np.random.randint(0,input_data_numpy.shape[0]/2)
    audio_fade = torchaudio.transforms.Fade(fade_in_len=fade_in, fade_out_len=fade_out, fade_shape=shape)(input_data_numpy)
    audio_fade = to_2_img(audio_fade,state)
    return audio_fade
#######################################################################
###############################   neg aug    #####################################
'''
制造负样本
'''
def flip(input_spectrogram):
    flip_spectrogram_1 = torch.flip(input_spectrogram[0],dims=[0])
    flip_spectrogram_2 = torch.flip(input_spectrogram[1],dims=[0])
    out1 = torch.unsqueeze(flip_spectrogram_1,dim=0)
    out2 = torch.unsqueeze(flip_spectrogram_2,dim=0)
    out = torch.cat((out1, out2), dim=0)
    return out
def plus2(input_spectrogram):
    out = input_spectrogram * 2
    return out
def double(input_spectrogram):
    spectrogram1 = input_spectrogram[0]
    spectrogram2 = input_spectrogram[1]
    mid1 = spectrogram1[:112,:]
    mid2 = spectrogram2[:112,:]
    spectrogram1[112:,:] = mid1
    spectrogram2[112:,:] = mid2
    spectrogram1 = torch.unsqueeze(spectrogram1, dim=0)
    spectrogram2 = torch.unsqueeze(spectrogram2, dim=0)
    out_spectrogram = torch.cat((spectrogram1, spectrogram2), dim=0)
    return out_spectrogram        
####################################    特征图    #################################
def log_mel_power(input_data_numpy, sample_rate=16000):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=224, n_fft=2048, hop_length=512, f_max=sample_rate/2, power=1.0, normalized=False)(input_data_numpy)
    mel_spectrogram = 20 * torch.log10(torch.clamp(mel_spectrogram, min=1e-5)) - 20
    log_mel_power = torch.clamp((mel_spectrogram + 100) / 100, 0.0, 1.0)
    return log_mel_power
def log_mel_energy(input_data_numpy, sample_rate=16000):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=224, n_fft=2048, hop_length=512, f_max=sample_rate/2, power=2.0, normalized=False)(input_data_numpy)
    mel_spectrogram = 20 * torch.log10(torch.clamp(mel_spectrogram, min=1e-5)) - 20
    log_mel_energy = torch.clamp((mel_spectrogram + 100) / 100, 0.0, 1.0)
    return log_mel_energy
def to_2_img(input_data_numpy, state=True):
    if state ==True:
        img = torch.zeros((2,224,224))
        img[0,:,:] = (log_mel_power(input_data_numpy))
        img[1,:,:] = (log_mel_energy(input_data_numpy))
    else:
        img = torch.zeros((2,224,313))
        img[0,:,:] = (log_mel_power(input_data_numpy))
        img[1,:,:] = (log_mel_energy(input_data_numpy))
    return img
###########################################################################
########################## MIX UP ######################
def mix_up(input1, input2, label1, label2):
    lamda = 0.4
    output_data = (1 - lamda) * input1 + lamda * input2
    output_label = (1 - lamda) * label1 + lamda * label2
    return output_data, output_label
#################################################################
################  Arcface loss  ###################
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=16, out_features=16, s=50.0, m=0.01, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features,in_features))  #为W矩阵
        nn.init.xavier_normal_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) # x = x'/ ||x'||2, Wj = Wj' / ||Wj'||2
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m #cos(θ+m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (label * phi) + ((1.0 - label) * cosine)
        output *= self.s
        output = nn.CrossEntropyLoss()(output,label)
        return output
###########################################################
##########################################################
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
    def forward(self, p, z):
        z = z.detach()
        p = F.normalize(p, p=2, dim=1)     #  x / ||x||2
        z = F.normalize(z, p=2, dim=1)
        return -(p * z).sum(dim=1).mean()
class projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.pro = nn.Sequential(
            nn.Linear(1280, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 1280),
            nn.BatchNorm1d(1280)
            )    
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        out = self.pro(x)
        return out
class predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(1280,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1280)
            )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    def forward(self,x):
        out = self.pre(x)
        return out
class contrastive_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = D()
        self.pre = predictor()
        self.pro = projector()
    def forward(self,x1,x2):
        z1 = self.pro(x1)
        z2 = self.pro(x2)
        
        p1 = self.pre(z1)
        p2 = self.pre(z2)
        
        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.
        return d1,d2
######################################################################
######################################################################
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256,16)
            )
    def forward(self,x):
        out = self.fc(x)
        return out
class time_decoder(nn.Module):
    def __init__(self):
        super(time_decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 16)
            )
    def forward(self, x):
        out = self.fc(x)
        return out
    
###################################### 
#######################################################################
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss,self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        distance_for_positive = ((anchor - positive).pow(2)).sum(1)
        distance_for_negative = ((anchor - negative).pow(2)).sum(1)
        losses = F.relu(distance_for_positive - distance_for_negative + self.margin)
        losses = losses.mean()
        return losses      