#!/usr/bin/env python3
import argparse
import librosa
from tqdm import tqdm
import io
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from pypeln import process as pr
import gzip
import h5py

parser = argparse.ArgumentParser()
# parser.add_argument('input_csv')
# parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-c', type=int, default=4)
parser.add_argument('-sr', type=int, default=22050)
parser.add_argument('-col',
                    default='filename',
                    type=str,
                    help='Column to search for audio files')
parser.add_argument('-cmn', default=False, action='store_true')
parser.add_argument('-cvn', default=False, action='store_true')
parser.add_argument('-winlen',
                    default=40,
                    type=float,
                    help='FFT duration in ms')
parser.add_argument('-hoplen',
                    default=20,
                    type=float,
                    help='hop duration in ms')

parser.add_argument('-n_mels', default=64, type=int)
ARGS = parser.parse_args()

what_type = 'validate'
weak_csv = '/home/ydc/wsed/target_sound_detection/data/flists/urban_sed_' + what_type +'_weak.tsv'
DF_weak = pd.read_csv(weak_csv, sep='\t',usecols=[0,1])  # only read first cols, allows to have messy csv

strong_csv = '/home/ydc/wsed/target_sound_detection/data/flists/urban_sed_' + what_type +'_strong.tsv'

print('weak_csv ',weak_csv)
print('strong_csv ',strong_csv)
DF_strong = pd.read_csv(strong_csv,sep='\t',usecols=[0,1,2,3])
MEL_ARGS = {
    'n_mels': ARGS.n_mels,
    'n_fft': 2048,
    'hop_length': int(ARGS.sr * ARGS.hoplen / 1000),
    'win_length': int(ARGS.sr * ARGS.winlen / 1000)
}

EPS = np.spacing(1)


def extract_feature(fname):
    """extract_feature
    Extracts a log mel spectrogram feature from a filename, currently supports two filetypes:
    1. Wave
    2. Gzipped wave
    :param fname: filepath to the file to extract
    """
    ext = Path(fname).suffix
    try:
        if ext == '.gz':
            with gzip.open(fname, 'rb') as gzipped_wav:
                y, sr = sf.read(io.BytesIO(gzipped_wav.read()),
                                dtype='float32')
                # Multiple channels, reduce
                if y.ndim == 2:
                    y = y.mean(1)
                y = librosa.resample(y, sr, ARGS.sr)
        elif ext in ('.wav', '.flac'):
            y, sr = sf.read(fname, dtype='float32')
            if y.ndim > 1:
                y = y.mean(1)
            y = librosa.resample(y, sr, ARGS.sr)
    except Exception as e:
        # Exception usually happens because some data has 6 channels , which librosa cant handle
        logging.error(e)
        logging.error(fname)
        raise
    lms_feature = np.log(librosa.feature.melspectrogram(y, **MEL_ARGS) + EPS).T
    return fname, lms_feature

frames_num = 501
num_freq_bin = 64
h5_name = '/data/ydc_data/urban_target_detection_' + what_type + '_add_neg.h5'
print(h5_name)
hf = h5py.File(h5_name, 'w')
hf.create_dataset(
    name='filename', 
    shape=(0,), 
    maxshape=(None,),
    dtype='S80')
hf.create_dataset(
    name='target_event', 
    shape=(0,), 
    maxshape=(None,),
    dtype='S80')
hf.create_dataset(
    name='mel_feature', 
    shape=(0, frames_num, num_freq_bin), 
    maxshape=(None, frames_num, num_freq_bin), 
    dtype=np.float32)
hf.create_dataset(
    name='time',
    shape=(0,10,2),
    maxshape=(None,10,2),
    dtype=np.float32
)
weak_filename = DF_weak['filename']
weak_label = DF_weak['event_labels']
n=0
for i,filename in enumerate(weak_filename):
    basename = Path(filename).name
    print(i,basename)
    strong_name = '/home/ydc/wsed/CDur/data/URBAN-SED/audio/'what_type +'/'+basename # validate/
    fname, lms_feature = extract_feature(strong_name)
    # print(fname,lms_feature.shape)
    # print(i,filename)
    event_labels = weak_label[i]
    ls_event = event_labels.split(',')
    new_ls_event = []
    for event in ls_event: # delete overlap item
        if event not in new_ls_event:
            new_ls_event.append(event)
    for event in new_ls_event:
        time_label = []
        for j,strong in enumerate(DF_strong['filename']):
            if strong_name == strong:
                if event == DF_strong['event_label'][j]:
                    st = DF_strong['onset'][j]
                    ed = DF_strong['offset'][j]
                    st = float(st)
                    ed = float(ed)
                    tmp = [st,ed]
                    time_label.append(tmp)
        hf['mel_feature'].resize((n + 1, frames_num, num_freq_bin))
        hf['mel_feature'][n] = lms_feature

        hf['filename'].resize((n+1,))
        hf['filename'][n] = basename.encode()

        hf['target_event'].resize((n+1,))
        hf['target_event'][n] = event.encode()

        hf['time'].resize((n+1,10,2))
        while len(time_label) < 10:
            time_label.append([-1,-1])
        assert len(time_label) == 10
        time_label = np.array(time_label)
        hf['time'][n] = time_label
        n += 1
print(n)



