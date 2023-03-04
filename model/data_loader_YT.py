# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from scipy.interpolate import interp1d


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
        self.interpolation_kind = 'nearest' 

        ### filename is h5
        # /{key}
        # TODO account for step_size != stack_size when generating video features
        #    /features (n_features, 1024)
        #    /heat-markers (100)
        
        # heat_markers need to be interpolated to n_features


        self.filename = '../PGL-SUM/data/datasets/'+ self.name + '/out.h5'
        self.splits_filename = ['../PGL-SUM/data/datasets/splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features, self.list_gtscores = [], []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            frame_features = np.array(hdf[video_name + '/features'])
            heat_markers = np.array(hdf[video_name + '/heat-markers'])
            interpolation_num = heat_markers.shape[0] # maybe + 1
            gtscore = interp1d(np.linspace(0, frame_features.shape[0]-1, interpolation_num), heat_markers, kind=self.interpolation_kind, assume_sorted=True)(np.arange(0, frame_features.shape[0]))

            self.list_frame_features.append(torch.Tensor(frame_features))
            self.list_gtscores.append(torch.Tensor(gtscore))

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]

        # if self.mode == 'test':
        #     return frame_features, gtscore, video_name
        # else:
        return frame_features, gtscore, video_name


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass
