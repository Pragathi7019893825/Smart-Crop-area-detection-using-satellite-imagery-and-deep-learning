from PIL import Image
import numpy as np
import os
p = 'train/100694_mask.png'
print('exists', os.path.exists(p))
arr = np.array(Image.open(p))
print('shape', arr.shape, 'dtype', arr.dtype)
if arr.ndim == 3:
    uniq = np.unique(arr.reshape(-1, arr.shape[2]), axis=0)
else:
    uniq = np.unique(arr)
print('unique', uniq.tolist()[:20])
print('min', arr.min(), 'max', arr.max())
print('channels', arr.ndim)
