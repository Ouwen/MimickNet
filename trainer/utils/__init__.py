from trainer.utils.dataset import MimickDataset
from trainer.utils.callbacks import *
from trainer.utils.losses import *

def mat2model(matfile_path, log_compress=True, clipping=False):
    matfile = sio.loadmat(matfile_path)
    iq = np.abs(matfile['iq'])
    if clipping:        
        iq = 20*np.log10(iq/iq.max())
        iq = np.clip(iq, -80, 0)
    elif log_compress:
        iq = np.log10(iq)
    iq = (iq-iq.min())/(iq.max() - iq.min())

    dtce = matfile['dtce']
    dtce = (dtce - dtce.min())/(dtce.max() - dtce.min())
    
    shape = iq.shape

    height = iq.shape[0]
    h_pad = 0
    if height % 16 != 0:
        h_pad = 16 - height % 16

    width = iq.shape[1]
    w_pad = 0
    if width % 16 != 0:
        w_pad = 16 - width % 16
    iq = np.pad(iq, [(0, h_pad), (0, w_pad)], 'reflect')
    iq = np.expand_dims(np.expand_dims(iq, axis=0), axis=-1)
    dtce = np.pad(dtce, [(0, h_pad), (0, w_pad)], 'reflect')
    dtce = np.expand_dims(np.expand_dims(dtce, axis=0), axis=-1)
    return iq, dtce, shape