# Import All Necessary Packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torchaudio
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
from tabulate import tabulate # tabulate print
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch.nn.functional as F
from pathlib import Path
import h5py
warnings.filterwarnings("ignore") # Ignore All Warnings
df = pd.read_csv("/home/ydc/wsed/target_sound_detection/data/UrbanSound8K/metadata/UrbanSound8K.csv")
df.head()

# wave = torchaudio.load("../input/urbansound8k/fold1/102106-3-0-0.wav")
# plt.plot(wave[0].t().numpy())
# print(wave[0].shape) # torch.Size([2, 72324]) 2 channels, 72324 sample_rate

class AudioDataset:
    def __init__(self, file_path, class_id):
        self.file_path = file_path
        self.class_id = class_id
        
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        path = self.file_path[idx]
        waveform, sr = torchaudio.load(path, normalization=True) # load audio ,可能是多通道数据
        audio_mono = torch.mean(waveform, dim=0, keepdim=True) # Convert sterio to mono
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000: # if sample_rate < 160000
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000] # else sample_rate 160000
        audio_mono=tempData
        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono) # (channel, n_mels, time) (1,128,801)
        # print('mel_specgram ',mel_specgram.shape)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std() # Noramalization
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono) # (channel, n_mfcc, time) (1,40,801)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std() # mfcc norm
        new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        return {
            "specgram": torch.tensor(new_feat[0].permute(1, 0), dtype=torch.float),
            "label": torch.tensor(self.class_id[idx], dtype=torch.long)
        }

# device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(data):
    specs = []
    labels = []
    for d in data:
        spec = d["specgram"].to(device)
        label = d["label"].to(device)
        specs.append(spec)
        labels.append(label)
    spec = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True, padding_value=0.)
    labels = torch.tensor(labels)
    return spec, labels


FILE_PATH = "/home/ydc/wsed/target_sound_detection/data/UrbanSound8K/audio/"

df = pd.read_csv("/home/ydc/wsed/target_sound_detection/data/UrbanSound8K/metadata/UrbanSound8K.csv")
files = df["slice_file_name"].values.tolist()
folder_fold = df["fold"].values
label = df["classID"].values.tolist()
path = [os.path.join(FILE_PATH + "fold" + str(folder) + "/" + file) for folder, file in zip(folder_fold, files)]

X_train, X_test, y_train, y_test = model_selection.train_test_split(path, label, random_state=42, test_size=0.3)

train_dataset = AudioDataset(file_path=X_train,class_id=y_train)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)

test_dataset = AudioDataset(file_path=X_test,class_id=y_test)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True, collate_fn=collate_fn)

class AudioLSTM(nn.Module):
    def __init__(self, n_feature=5, out_feature=5, n_hidden=256, n_layers=2, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature
        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(n_hidden), int(n_hidden/2))
        self.fc2 = nn.Linear(int(n_hidden/2), out_feature)

    def forward(self, x, hidden):
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)
        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)
        out = self.fc1(out)
        # print('out1 ',out.shape)
        out = self.fc2(out[:, -1, :])
        # return the final output and the hidden state
        return out, l_hidden
    
    def extract(self,x,hidden):
        l_out, l_hidden = self.lstm(x, hidden)
        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)
        # out.shape (batch, out_feature)
        out = self.fc1(out)
        return out[:,-1,:]

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        return hidden


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
class AudioCNN(nn.Module):
    def __init__(self, classes_num):
        super(AudioCNN, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512,128,bias=True)
        self.fc = nn.Linear(128, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        # [128, 801, 168] --> [128,1,801,168]
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') # 128,64,400,84
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg') # 128,128,200,42
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # 128,256,100,21
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') # 128,512,50,10
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes) # 128,512,50
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps) 128,512
        x = self.fc1(x) # 128,128
        output = self.fc(x) # 128,10
        return output

    def extract(self,input):
        '''Input: (batch_size, times_steps, freq_bins)'''
        x = input[:, None, :, :]
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc1(x) # 128,128
        return x

   
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
def save_model(state, filename):
    torch.save(state, filename)
    print("-> Model Saved")
def train(data_loader, model, epoch, optimizer, device):
    losses = []
    accuracies = []
    labels = []
    preds = []
    model.train()
    loop = tqdm(data_loader) # for progress bar
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device)
        target = target.to(device)
        model.zero_grad()
        output, hidden_state = model(data, model.init_hidden(128))
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        probs = torch.softmax(output, dim=1)
        winners = probs.argmax(dim=1)
        corrects = (winners == target)
        accuracy = corrects.sum().float() / float(target.size(0))
        accuracies.append(accuracy)
        labels += torch.flatten(target).cpu()
        preds += torch.flatten(winners).cpu()
        loop.set_description(f"EPOCH: {epoch} | ITERATION : {batch_idx}/{len(data_loader)} | LOSS: {loss.item()} | ACCURACY: {accuracy}")
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = sum(losses) / len(losses)
    avg_train_accuracy = sum(accuracies) / len(accuracies)
    report = metrics.classification_report(torch.tensor(labels).numpy(), torch.tensor(preds).numpy())
    print(report)
    return avg_train_loss, avg_train_accuracy

def test(data_loader, model, optimizer, device):
    model.eval()
    accs = []
    preds = []
    labels = []
    test_accuracies = []
    with torch.no_grad():
        loop = tqdm(data_loader) # Test progress bar
        for batch_idx, (data, target) in enumerate(loop):
            data = data.to(device)
            target = target.to(device)
            output, hidden_state = model(data, model.init_hidden(128))
            probs = torch.softmax(output, dim=1)
            winners = probs.argmax(dim=1)
            corrects = (winners == target)
            accuracy = corrects.sum().float() / float(target.size(0))
            test_accuracies.append(accuracy)
            labels += torch.flatten(target).cpu()
            preds += torch.flatten(winners).cpu()
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    return avg_test_acc

def train_cnn(data_loader, model, epoch, optimizer, device):
    losses = []
    accuracies = []
    labels = []
    preds = []
    model.train()
    loop = tqdm(data_loader) # for progress bar
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device)
        target = target.to(device)
        model.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        probs = torch.softmax(output, dim=1)
        winners = probs.argmax(dim=1)
        corrects = (winners == target)
        accuracy = corrects.sum().float() / float(target.size(0))
        accuracies.append(accuracy)
        labels += torch.flatten(target).cpu()
        preds += torch.flatten(winners).cpu()
        loop.set_description(f"EPOCH: {epoch} | ITERATION : {batch_idx}/{len(data_loader)} | LOSS: {loss.item()} | ACCURACY: {accuracy}")
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = sum(losses) / len(losses)
    avg_train_accuracy = sum(accuracies) / len(accuracies)
    report = metrics.classification_report(torch.tensor(labels).numpy(), torch.tensor(preds).numpy())
    print(report)
    return avg_train_loss, avg_train_accuracy

def test_cnn(data_loader, model, optimizer, device):
    model.eval()
    accs = []
    preds = []
    labels = []
    test_accuracies = []
    with torch.no_grad():
        loop = tqdm(data_loader) # Test progress bar
        for batch_idx, (data, target) in enumerate(loop):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            winners = probs.argmax(dim=1)
            corrects = (winners == target)
            accuracy = corrects.sum().float() / float(target.size(0))
            test_accuracies.append(accuracy)
            labels += torch.flatten(target).cpu()
            preds += torch.flatten(winners).cpu()
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    return avg_test_acc

EPOCH = 50
OUT_FEATURE = 10 # class
PATIENCE = 5
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AudioLSTM(n_feature=168, out_feature=OUT_FEATURE).to(device)
    model = AudioCNN(10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=PATIENCE)
    
    best_train_acc, best_epoch = 0, 0 # update acc and epoch
    
    for epoch in range(EPOCH):
        avg_train_loss, avg_train_acc = train_cnn(train_loader, model, epoch, optimizer, device)
        avg_test_acc = test_cnn(test_loader, model, optimizer, device)
        scheduler.step(avg_train_acc)
        if avg_train_acc > best_train_acc:
            best_train_acc = avg_train_acc
            best_epoch = epoch
            filename = f"cnn_best_model_at_epoch_{best_epoch}.pth.tar"
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_model(checkpoint, filename)
        
        table = [
            ["avg_train_loss", avg_train_loss], ["avg_train_accuracy", avg_train_acc],
            ["best_train_acc", best_train_acc], ["best_epoch", best_epoch]
        ]
        print(tabulate(table)) # tabulate View
        test_table = [
            ["Avg test accuracy", avg_test_acc]
        ]
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/test', avg_test_acc, epoch)
        print(tabulate(test_table)) # tabulate View

def get_embedding():
    file_write_obj1 = open("spk_embed.128.txt", 'w')
    df = pd.read_csv("/home/ydc/wsed/target_sound_detection/data/UrbanSound8K/metadata/UrbanSound8K.csv")
    files = df["slice_file_name"].values.tolist()
    folder_fold = df["fold"].values
    label = df["classID"].values.tolist()
    path = [os.path.join(FILE_PATH + "fold" + str(folder) + "/" + file) for folder, file in zip(folder_fold, files)]
    for i in range(len(path)):
        waveform, sr = torchaudio.load(path[i])
        print(i, path[i])
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000:
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000]
        audio_mono=tempData
        mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        
        data = torch.utils.data.DataLoader(new_feat.permute(0, 2, 1))
        new = torch.load("/home/ydc/wsed/target_sound_detection/src/cnn_best_model_at_epoch_45.pth.tar", map_location=torch.device("cpu"))["state_dict"]
        model = AudioCNN(OUT_FEATURE)
        model.load_state_dict(new)
        model.eval().cpu()
        with torch.no_grad():
            for x in data:
                x = x.to("cpu")
                output = model.extract(x)
                output = output.numpy()
                basename = Path(path[i]).name
                # ans_ =str(basename,encoding="ascii")
                file_write_obj1.write(basename+'\t')
                ft = ''
                for t in output[0,:]:
                    ft = ft+str(t)+' '
                file_write_obj1.write(ft)
                file_write_obj1.write('\n')
    file_write_obj1.close()

def get_mel_mfcc():
    # h5_name = '/home/ydc/wsed/target_sound_detection/data/features/mel_path.h5'
    # print(h5_name)
    # hf = h5py.File(h5_name, 'w')
    # hf.create_dataset(
    #     name='filename', 
    #     shape=(0,), 
    #     maxshape=(None,),
    #     dtype='S80')
    # hf.create_dataset(
    #     name='file_path', 
    #     shape=(0,), 
    #     maxshape=(None,),
    #     dtype='S160')
    df = pd.read_csv("/home/ydc/wsed/target_sound_detection/data/UrbanSound8K/metadata/UrbanSound8K.csv")
    files = df["slice_file_name"].values.tolist()
    folder_fold = df["fold"].values
    label = df["classID"].values.tolist()
    path = [os.path.join(FILE_PATH + "fold" + str(folder) + "/" + file) for folder, file in zip(folder_fold, files)]
    n = 0
    for i in range(len(path)):
        waveform, sr = torchaudio.load(path[i])
        print('sr ',sr)
        assert 1==2
        print(i, path[i])
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000:
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000]
        audio_mono=tempData
        output = audio_mono.numpy()
        basename = Path(path[i]).name
        # ans_ =str(basename,encoding="ascii")
        # mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
        # mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        # mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
        # mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        # new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        # new_feat = new_feat.permute(0, 2, 1).squeeze()
        hf['filename'].resize((n+1,))
        hf['filename'][n] = basename.encode()

        hf['file_path'].resize((n+1,))
        hf['file_path'][n] = path[i].encode()
        n += 1

    # file_write_obj1.close()
if __name__ == "__main__":
    get_mel_mfcc() # Run function