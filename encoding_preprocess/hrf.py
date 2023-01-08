import scipy.io as scio
import h5py
import hdf5storage as hdf5
import numpy as np
from nilearn import glm
import os

zs = lambda v: (v-v.mean(0))/v.std(0)

def convolve(in_path, time_root, out_path, hrf, ref_length):
    """
    inputs:
        in_path: embedding path
        time_root: root for time_align files, which contain the onset and offset time of each word
        out_path: path to save convolved features
        hrf: spm hrf, used to convolve embeddings
        ref_length: cut the useless TRs in the end, because the fMRI collection keeped running after 
            the end of the story, you can simply ignore this here and cut the extra fmri TRs.
    """
    for i in range(1, 61):
        data = scio.loadmat(in_path+'story_%d_word2vec.mat'%i)
        data = data['data'] #1044，300
        word_time = scio.loadmat(time_root+'story_%d_word_time.mat'%i)
        word_time = word_time['end'] #1044
        length = int(word_time[0][-1]*100)
        time_series = np.zeros([length, data.shape[1]])
        t = 0
        for j in range(length):
            if j == int(word_time[0][t]*100):
                time_series[j] = data[t]
                while(j == int(word_time[0][t]*100)):
                    t += 1
                    if t == data.shape[0]:
                        break
        conv_series = []
        for j in range(data.shape[1]):
            conv_series.append(np.convolve(hrf, time_series[:,j]))
        conv_series = np.stack(conv_series).T
        conv_series = conv_series[:length]
        conv_series_ds = [conv_series[j] for j in range(0, length, 71)]
        conv_series_ds = np.array(conv_series_ds) #587, 300
        
        word_feature = zs(conv_series_ds[19:ref_length[i-1]+19]) # z-score, dump the first 19 TRs becuase they were blank
        tmp = {'word_feature':word_feature.astype('float32')}
        hdf5.writes(tmp, out_path+'/story_%d.mat'%i, matlab_compatible=True)

def load_ref_TRs():
    file_root = './'
    res = []
    for i in range(1, 61): # 60 stories
        data = h5py.File(file_root+'story_%d.mat'%i, 'r')
        res.append(data['fmri_response'].shape[0])
    return res

if __name__ == "__main__":
    time_root = './'
    hrf = glm.first_level.spm_hrf(0.71, 71)
    ref_length = load_ref_TRs()

    # gpt2-hf
    in_root = './'
    out_root = './'
    # for i in range(1, 13):
    for i in range(1, 2):
        in_path = in_root + '/'
        out_path = out_root + '/word2vec/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        convolve(in_path, time_root, out_path, hrf, ref_length)
