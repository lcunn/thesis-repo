import os
import sys
import time
import logging
import datetime
import glob
import random
import argparse
from typing import List, Dict

import numpy as np
import torch
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sms.exp1.data.formatter import InputFormatter
from sms.src.synthetic_data.note_arr_mod import NoteArrayModifier

class OneBarChunkDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        format_config: Dict[str, bool],
        split: str = 'train',
        split_ratio: float = 0.8,
        use_transposition: bool = False,
        return_negative_example: bool = False,
        neg_enhance = False
        ):

        self.formatter = InputFormatter(**format_config)
        self.use_transposition = use_transposition
        self.return_negative_example = return_negative_example
        
        if neg_enhance:
            self.negative_sample = self.low_edit_distance_sample
        else:
            self.negative_sample = self.new_sample

        self.augmentation_dict = {
            1: 'use_transposition',
            2: 'use_shift_selected_notes_pitch',
            3: 'use_change_note_durations',
            4: 'use_delete_notes',
            5: 'use_insert_notes'
        }

        self.loaded_data = []
        for path in data_paths:
            data = torch.load(path)
            self.loaded_data.append(torch.from_numpy(arr) for arr in data.values())

        # Split the data
        split_idx = int(len(self.loaded_data) * split_ratio)
        if split == 'train':
            self.loaded_data = self.loaded_data[:split_idx]
        elif split == 'val':
            self.loaded_data = self.loaded_data[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.loaded_data)
    
    def new_sample(self, idx) -> int:
        """
        Samples an idx from loaded_data other than the input idx.
        """
        new_idx = idx
        while new_idx == idx:
            new_idx = random.randint(0, len(self.loaded_data) - 1)
        return new_idx
        
    def negative_enhance_sample(self, idx):
        """
        Uses rejection sampling to sample a sufficiently negative sample from the dataset.
        Calculates a rough approximation of similarity between the anchor by taking difference between the quantized relative bars.
        """
        formatter = InputFormatter(make_relative_pitch=True, quantize=True)
        anchor = formatter(self.loaded_data[idx])
        new_idx = self.new_sample(idx)
        
        negative = self.loaded_data[new_idx]
        negative_quantized = self.formatter(negative)
        return anchor_quantized, negative_quantized
        

    def __getitem__(self, idx):
        chunk = self.loaded_data[idx]
        # randomly pick one augmentation
        if self.use_transposition:
            idx = np.random.randint(1, 5)
        else:
            idx = np.random.randint(2, 5)
        augmentation = {self.augmentation_dict[idx]: True}
        # apply the augmentation
        modifier = NoteArrayModifier()
        augmented_chunk = modifier(chunk, augmentation)

        if self.negative_example:
            negative_idx = 
        
        return self.formatter(chunk), self.formatter(augmented_chunk)

def sequence_collate_fn(batch):
    # Separate anchors and positives
    anchors = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]
    
    # Pad sequences
    anchors_padded = pad_sequence(anchors, batch_first=True, padding_value=0)
    positives_padded = pad_sequence(positives, batch_first=True, padding_value=0)
    
    # Create masks
    anchor_lengths = torch.LongTensor([len(x) for x in anchors])
    positive_lengths = torch.LongTensor([len(x) for x in positives])
    
    anchor_mask = (torch.arange(anchors_padded.size(1))[None, :] < anchor_lengths[:, None]).float()
    positive_mask = (torch.arange(positives_padded.size(1))[None, :] < positive_lengths[:, None]).float()
    
    return {
        'anchors': anchors_padded,
        'positives': positives_padded,
        'anchor_mask': anchor_mask,
        'positive_mask': positive_mask,
        'anchor_lengths': anchor_lengths,
        'positive_lengths': positive_lengths
    }
    
def get_dataloader(
        data_paths, 
        format_config, 
        use_transposition, 
        neg_enhance, 
        split,
        split_ratio,
        batch_size, 
        num_workers,
        use_sequence_collate_fn=False,
        shuffle=True
        ) -> DataLoader:
    
    dataset = OneBarChunkDataset(
        data_paths, 
        format_config, 
        use_transposition=use_transposition, 
        neg_enhance=neg_enhance,
        split=split,
        split_ratio=split_ratio
        )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle,
        collate_fn=sequence_collate_fn if use_sequence_collate_fn else None
        )
