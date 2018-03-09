import numpy as np
import os, sys
from skimage import io

directories = {'no_train' : 'no_THA_train',
                'yes_train' : 'yes_THA_train',
                'no_test' : 'no_THA_test',
                'yes_test' : 'yes_THA_test'}

data = []
for dir_name in ['no_train', 'yes_train']:
    samples = os.listdir(directories[dir_name])
    for sample in samples:
        if not sample.startswith('.'): # avoid .DS_Store
            x = io.imread(os.path.join(directories[dir_name], sample))
            x = np.resize(x, (x.shape[0], x.shape[1], 3))
            y = np.mean(x, axis=(0, 1))
            data.append(y)

data = np.array(data)
print(np.mean(data, axis=0))
print(np.std(data, axis=0))