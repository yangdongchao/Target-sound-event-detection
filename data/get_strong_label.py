import h5py
import pandas as pd
import numpy as np
save_name = []
save_onset = []
save_offset = []
save_event = []
with h5py.File('/data/ydc_data/urban_target_detection_test_add_neg.h5', 'r') as hf:
    X = hf['mel_feature'][:].astype(np.float32)
    y = hf['time'][:].astype(np.float32)
    target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
    filename = np.array([filename.decode() for filename in hf['filename'][:]])
    for i in range(filename.shape[0]):
        print(filename[i])
        for j in range(10):
            if y[i,j,0] == -1:
                break
            save_name.append(filename[i])
            save_onset.append(y[i,j,0])
            save_offset.append(y[i,j,1])
            save_event.append(target_event[i])

    dict = {'filename': save_name, 'onset': save_onset, 'offset': save_offset, 'event_label': save_event}
    df = pd.DataFrame(dict)
    df.to_csv('strong_test_add_neg.tsv',index=False,sep='\t')

