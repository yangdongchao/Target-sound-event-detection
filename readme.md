### first step ---> data process

if a mixture audio include N envents, so that we can generate N samples.
sample style： filename_event target_event mel_feature time
eg.  soundscape_test_bimodal0_gun_shot.wav gun_shot [[0,1],[1,1]]  [[],[]]


urbanSound8K
sample style： filename feature 
eg. 100263-2-0-117.wav []

In our experiments, we randomly choose embedding according to class name.

### How to run code
bash bash/tsd.sh


### Join train
please run bash/jtsd.sh

### TO do list
We only realease the fully supervised training code, the weakly training code is similar with this. If you need it, please let me known.

### reference
https://github.com/RicherMans/CDur
https://www.kaggle.com/adinishad/urbansound-classification-with-pytorch-and-fun
https://github.com/qiuqiangkong/audioset_tagging_cnn
