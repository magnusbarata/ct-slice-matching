import pydicom as dcm
import numpy as np
import keras
from skimage.transform import resize

class TripletGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, num_slices=None, batch_size=32, dim=(512, 512), n_channels=1, shuffle=True, class_range=3, generate_siamese=False, strict_sampling=0.3):
        self.list_IDs = list_IDs
        self.num_slices = num_slices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.n_channels = n_channels
        self.class_range = class_range
        self.gen_siamese = generate_siamese
        self.gamma = strict_sampling

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.list_IDs) // self.batch_size)

    def __getitem__(self, idx):
        b_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        if self.num_slices is None:
            #TODO
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            for i, index  in enumerate(b_indices):
                fquery = self.list_IDs[index]
                X[i,] = self.getPixelData(fquery)

            return X
        elif self.gen_siamese:
            X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
            X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = [0] * self.batch_size
            for i, index  in enumerate(b_indices):
                fquery = self.list_IDs[index]
                fneg, fpos, _ = self.get_fpos_fneg(fquery, self.num_slices[index])
                X1[i,] = self.getPixelData(fquery)

                if np.random.rand(1) < 0.5:
                    X2[i,] = self.getPixelData(fneg)
                    y[i] = 0
                else:
                    X2[i,] = self.getPixelData(fpos)
                    y[i] = 1

            return [X1, X2], y
        else:
            X = np.empty((3 * self.batch_size, *self.dim, self.n_channels))
            y = [0] * 3 * self.batch_size
            for i, index  in enumerate(b_indices):
                fquery = self.list_IDs[index]
                fneg, fpos, slice_num = self.get_fpos_fneg(fquery, self.num_slices[index])

                X[i,] = self.getPixelData(fquery)
                X[i+self.batch_size,] = self.getPixelData(fneg)
                X[i+2*self.batch_size,] = self.getPixelData(fpos)

                y[i] = slice_num[0]
                y[i+self.batch_size] = slice_num[1]
                y[i+2*self.batch_size] = slice_num[2]

            return X, y

    def get_fpos_fneg(self, fquery, max_index):
        base_path = fquery.split('/')[:-1]
        anchor_idx = int(fquery.split('/')[-1][:-4])
        if self.gamma > 0.0: self.class_range = 1

        if anchor_idx - self.class_range < 1: low_i = 1
        else: low_i = anchor_idx - self.class_range

        if anchor_idx + self.class_range > max_index: high_i = max_index
        else: high_i = anchor_idx + self.class_range

        all_indices = np.arange(1, max_index+1)
        all_indices = all_indices[all_indices != anchor_idx]
        if np.random.rand(1) < self.gamma:
            all_indices = all_indices[(all_indices >= anchor_idx-2) & (all_indices <= anchor_idx+2)]

        neg_mask = np.where((all_indices < low_i) | (all_indices > high_i), True, False)
        neg_idx = int(np.random.choice(all_indices[neg_mask], 1))
        pos_idx = int(np.random.choice(all_indices[~neg_mask], 1))

        fneg = '/'.join(base_path) + '/%08d' % (neg_idx) + '.DCM'
        fpos = '/'.join(base_path) + '/%08d' % (pos_idx) + '.DCM'
        return fneg, fpos, [anchor_idx, neg_idx, pos_idx]

    def getPixelData(self, fname):
        ds = dcm.dcmread(fname)

        pixel_array = ds.pixel_array.astype('float64') - ds.RescaleIntercept
        if self.dim != pixel_array.shape:
            pixel_array = resize(pixel_array, self.dim)

        return np.expand_dims(pixel_array, -1)
