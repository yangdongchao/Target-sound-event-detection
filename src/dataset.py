from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scipy
from h5py import File
from tqdm import tqdm
import torch.utils.data as tdata
import os
import h5py
import torchaudio
import random
event_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

event_to_id = {label : i for i, label in enumerate(event_labels)}
id_to_event = {i: label for i,label in enumerate(event_labels)}
def read_spk_emb_file(spk_emb_file_path):
    print('get spk_id_dict and spk_emb_dict')
    spk_id_dict = {}
    spk_emb_dict = {}
    with open(spk_emb_file_path, 'r') as file:
        for line in file:
            temp_line = line.strip().split('\t')
            file_id = os.path.basename(temp_line[0])
            emb = np.array(temp_line[1].split(' ')).astype(np.float)
            spk_id = int(file_id.split('-')[1])
            spk_id_label = id_to_event[spk_id]
            spk_emb_dict[file_id] = emb
            if spk_id_label in spk_id_dict:
                spk_id_dict[spk_id_label].append(file_id)
            else:
                spk_id_dict[spk_id_label] = [file_id]
    return spk_emb_dict, spk_id_dict

def read_spk_emb_file_by_h5(spk_emb_file_path):
    print('get spk_id_dict and spk_emb_dict')
    spk_id_dict = {}
    spk_emb_dict = {}
    mel_mfcc = h5py.File(spk_emb_file_path, 'r') # libver='latest', swmr=True
    file_name = np.array([filename.decode() for filename in mel_mfcc['filename'][:]])
    file_path = np.array([file_path.decode() for file_path in mel_mfcc['file_path'][:]])
    for i in range(file_name.shape[0]):
        file_id = file_name[i]
        emb = file_path[i]
        spk_id = int(file_id.split('-')[1])
        spk_id_label = id_to_event[spk_id]
        spk_emb_dict[file_id] = emb
        if spk_id_label in spk_id_dict:
            spk_id_dict[spk_id_label].append(file_id)
        else:
            spk_id_dict[spk_id_label] = [file_id]
    
    return spk_emb_dict,spk_id_dict
def time_to_frame(tim):
    return int(tim/0.04)  # 10/250
class HDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_id_dict = read_spk_emb_file(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        data = self.X[index]
        fname = self.filename[index]
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_id_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.zeros(250) # 501 --> pooling 一次 到 250
        for i in range(10):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(250,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event

class HDF5Dataset_join(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_join, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_id_dict = read_spk_emb_file_by_h5(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len
    
    def get_fea(self,path_):
        waveform, sr = torchaudio.load(path_)
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000:
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000]
        audio_mono=tempData
        output = audio_mono.numpy()
        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        new_feat = new_feat.permute(0, 2, 1).squeeze()
        return new_feat
    def __getitem__(self, index): 
        data = self.X[index]
        fname = self.filename[index]
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_id_dict[target_event], 2)
        k8_path = self.spk_emb_dict[embed_file_list[0]]
        # embedding = self.spk_emb_dict[embed_file_list[0]]
        embedding = self.get_fea(k8_path)
        embed_label = np.zeros(10)
        embed_label_index = int(embed_file_list[0].split('-')[1])
        # print('embedding ',embedding.shape)
        embed_label[embed_label_index] = 1
        frame_level_label = np.zeros(250) # 501 --> pooling 一次 到 250
        for i in range(10):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(250,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        embed_label = torch.as_tensor(embed_label).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding,embed_label, fname, target_event

def getdataloader(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_join(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_join(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist, batch_first=batch_first, padding_value=padding_value)
    return padded_seq

def sequential_collate(batches):
    seqs = []
    for data_seq in zip(*batches):
        # print('data_seq[0] ',data_seq[0].shape)
        if isinstance(data_seq[0],
                      (torch.Tensor)):  # is tensor, then pad
            data_seq = pad(data_seq)
        elif type(data_seq[0]) is list or type(
                data_seq[0]) is tuple:  # is label or something, do not pad
            data_seq = torch.as_tensor(data_seq)
        seqs.append(data_seq)
    return seqs
