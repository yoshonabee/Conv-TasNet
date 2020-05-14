# Created on 2018/12
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import librosa


class AudioDataset(Dataset):

    def __init__(self, audio_json, sample_rate=8000, segment_length=4.0):
        """
        Args:
            audio_json: json file including audio_id, mixture_path, vocal_path, accompaniment_path, audio_length(s)
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)

        format of audio_json:
            audio_json = {
                <audio_id>:
                    {
                        mixture: <mixture path>,
                        vocal: <vocal path>,
                        accompaniment: <accompaniment path>,
                        length: (int)
                    }
            }
        """

        self.audio_json = audio_json
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.data = self.load_data(audio_json, sample_rate, segment_length)

    @staticmethod
    def load_data(audio_json, sample_rate, segment_length):
        audios = json.load(open(audio_json, 'r'))
        print(f"total {len(audios)} audios")

        segment_frames = int(segment_length * sample_rate)  # 4s * 8000/s = 32000 samples

        audios = sorted(audios.items(), key=lambda x: -int(x[1]['length']))
        data = []

        for audio_name, audio in audios:
            mixture_path = audio['mixture']
            vocal_path = audio['vocal']
            accompaniment_path = audio['accompaniment']

            mixture, _ = librosa.load(mixture_path, sr=sample_rate)
            vocal, _ = librosa.load(vocal_path, sr=sample_rate)
            accompaniment, _ = librosa.load(accompaniment_path, sr=sample_rate)

            s = np.dstack((vocal, accompaniment))[0]

            for i in range(0, mixture.shape[-1] - 1, segment_frames):
                data.append([mixture[i:i+segment_frames], s[i:i+segment_frames]])

        data = np.array(data)
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class AudioDataLoader(DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    mixtures = [b[0] for b in batch]
    sources = [b[1] for b in batch]

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    return mixtures_pad, ilens, sources_pad


# Eval data part
from preprocess import preprocess_one_dir

class EvalDataset(Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames


# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    mix_path, s1_path, s2_path, sample_rate, segment_len = batch
    # print(mix_infos)
    # print()
    # print(s1_infos)
    # print()
    # print(s2_infos)
    # print("\n\n\n\n\n")
    # for each utterance
    # for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        # mix_path = mix_info[0]
        # s1_path = s1_info[0]
        # s2_path = s2_info[0]
        # assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
    mix, _ = librosa.load(mix_path, sr=sample_rate)
    s1, _ = librosa.load(s1_path, sr=sample_rate)
    s2, _ = librosa.load(s2_path, sr=sample_rate)
    # merge s1 and s2
    # print(s1_path, s1.shape, s2_path, s2.shape)
    s = np.dstack((s1, s2))[0]  # T x C, C = 2
    utt_len = mix.shape[-1]
    if segment_len >= 0:
        # segment
        for i in range(0, utt_len - 1, segment_len):
            mixtures.append(mix[i:i+segment_len])
            sources.append(s[i:i+segment_len])
            if len(mixtures) == 4:
                break
    else:  # full utterance
        mixtures.append(mix)
        sources.append(s)

    return mixtures, sources


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
