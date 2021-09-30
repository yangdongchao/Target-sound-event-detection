from itertools import zip_longest
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
DEBUG=0
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]


class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / (time_decision.sum(
            self.pooldim)+1e-7)

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

        self.init_weight()
        
    def init_weight(self):
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
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14(nn.Module):
    def __init__(self):
        super(Cnn14, self).__init__()        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
 
    def forward(self, input_, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.conv_block1(input_, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 2), pool_type='avg')
        return x

class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)

class ResPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.linPool = LinearSoftPool(pooldim=1)

class AutoExpPool(nn.Module):
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.full((outputdim, ), 1))
        self.pooldim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        return (logits * torch.exp(scaled)).sum(
            self.pooldim) / torch.exp(scaled).sum(self.pooldim)


class SoftPool(nn.Module):
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        weight = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * weight, dim=self.dim)  # B x C


class ExtAttentionPool(nn.Module):
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.attention = nn.Linear(inputdim, outputdim)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.activ = nn.Softmax(dim=self.pooldim)

    def forward(self, logits, decision):
        # Logits of shape (B, T, D), decision of shape (B, T, C)
        w_x = self.activ(self.attention(logits) / self.outputdim)
        h = (logits.permute(0, 2, 1).contiguous().unsqueeze(-2) *
             w_x.unsqueeze(-1)).flatten(-2).contiguous()
        return torch.sum(h, self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)

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
        return x,output

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

def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'expalpha':
        return AutoExpPool(outputdim=kwargs['outputdim'], pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])
class conv1d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding='VALID', dilation=1):
        super(conv1d, self).__init__()
        if padding == 'VALID':
            dconv_pad = 0
        elif padding == 'SAME':
            dconv_pad = dilation * ((kernel_size - 1) // 2)
        else:
            raise ValueError("Padding Mode Error!")
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=dconv_pad)
        self.act = nn.ReLU()
        self.init_layer(self.conv)

    def init_layer(self, layer, nonlinearity='relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out
class Fusion(nn.Module):
    def __init__(self, inputdim,n_fac):
        super().__init__()
        self.fuse_layer1 = conv1d(inputdim, inputdim*n_fac,1)
        self.fuse_layer2 = conv1d(inputdim, inputdim*n_fac,1)
        self.avg_pool = nn.AvgPool1d(n_fac, stride=n_fac) # 沿着最后一个维度进行pooling

    def forward(self,embedding,mix_embed):
        embedding = embedding.permute(0,2,1)
        fuse1_out = self.fuse_layer1(embedding) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse1_out = fuse1_out.permute(0,2,1)

        mix_embed = mix_embed.permute(0,2,1)
        fuse2_out = self.fuse_layer2(mix_embed) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse2_out = fuse2_out.permute(0,2,1)
        as_embs = torch.mul(fuse1_out, fuse2_out) # 相乘 [2, 501, 2560]
        # (10, 501, 512)
        as_embs = self.avg_pool(as_embs) # [2, 501, 512] 相当于 2560//5
        return as_embs

class CDur_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.fusion = Fusion(128,4)
        self.fc = nn.Linear(256,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = self.fusion(embedding,x) # no fusion
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class Join(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,**kwargs):
        super().__init__()
        self.encoder = AudioCNN(10)
        self.detection = CDur(inputdim,outputdim)
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['state_dict'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
    
    def forward(self,x,ref):
        embedding,logit = self.encoder(ref)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,logit

class Join_fusion(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,**kwargs):
        super().__init__()
        self.encoder = AudioCNN(10)
        self.detection = CDur_fusion(inputdim,outputdim)
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['state_dict'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
    
    def forward(self,x,ref):
        embedding,logit = self.encoder(ref)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = self.detection.fusion(embedding,x)
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,logit