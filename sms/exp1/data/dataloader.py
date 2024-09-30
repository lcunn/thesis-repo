import os
import sys
import time
import logging
import datetime
import glob
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sms.src.synthetic_data.formatter import InputFormatter
from sms.src.synthetic_data.note_arr_mod import NoteArrayModifier

class OneBarChunkDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        formatter_config: Dict[str, bool],
        mode: str = 'pretrain',
        use_transposition: bool = False,
        use_negative_enhance: bool = False
        ):

        if mode not in ['pretrain', 'finetune']:
            raise ValueError("Mode must be either 'pretrain' or 'finetune'")

        self.formatter = InputFormatter(**formatter_config)
        self.use_transposition = use_transposition
        self.return_negative_example = mode == 'finetune'
        self.use_negative_enhance = use_negative_enhance
        self.modifier = NoteArrayModifier()

        self.augmentation_dict = {
            1: 'use_transposition',
            2: 'use_shift_selected_notes_pitch',
            3: 'use_change_note_durations',
            4: 'use_delete_notes',
            5: 'use_insert_notes'
        }

        # load data, which should be a list of np arrays, and convert to tensors
        data = torch.load(data_path)
        self.loaded_data = [torch.from_numpy(arr) for arr in data]

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
        
    def negative_enhance_sample(self, idx, threshold: float = 200):
        """
        Uses rejection sampling to sample a sufficiently negative sample from the dataset.
        Calculates a rough approximation of similarity between the anchor by taking the L1 difference between the quantized relative bars.
        """
        formatter = InputFormatter(make_relative_pitch=True, quantize=True)
        anchor = formatter(self.loaded_data[idx])
        distance = 0
        while distance < threshold:
            new_idx = self.new_sample(idx)
            negative = formatter(self.loaded_data[new_idx])
            distance = np.linalg.norm(anchor - negative, ord=1)
        return new_idx

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

        if self.return_negative_example:
            if self.use_negative_enhance:    
                negative_idx = self.negative_enhance_sample(idx)
            else:
                negative_idx = self.new_sample(idx)
            negative_chunk = self.loaded_data[negative_idx]
            return self.formatter(chunk).copy(), self.formatter(augmented_chunk).copy(), self.formatter(negative_chunk).copy()
        else:
            return self.formatter(chunk).copy(), self.formatter(augmented_chunk).copy()

# def sequence_collate_fn(batch):
#     # Separate anchors and positives
#     anchors = [item['anchor'] for item in batch]
#     positives = [item['positive'] for item in batch]
    
#     # Pad sequences
#     anchors_padded = pad_sequence(anchors, batch_first=True, padding_value=0)
#     positives_padded = pad_sequence(positives, batch_first=True, padding_value=0)
    
#     # Create masks
#     anchor_lengths = torch.LongTensor([len(x) for x in anchors])
#     positive_lengths = torch.LongTensor([len(x) for x in positives])
    
#     anchor_mask = (torch.arange(anchors_padded.size(1))[None, :] < anchor_lengths[:, None]).float()
#     positive_mask = (torch.arange(positives_padded.size(1))[None, :] < positive_lengths[:, None]).float()
    
#     return {
#         'anchors': anchors_padded,
#         'positives': positives_padded,
#         'anchor_mask': anchor_mask,
#         'positive_mask': positive_mask,
#         'anchor_lengths': anchor_lengths,
#         'positive_lengths': positive_lengths
#     }

def produce_train_test_data(data_paths: List[str], train_dest: str, val_dest: str, split_ratio: float = 0.8) -> None:
    """
    Splits the dataset into training and validation sets and saves them to the specified destinations.
    """
    # load all data
    all_data = []
    for path in data_paths:
        data = torch.load(path)
        # we load dictionaries
        all_data.extend(list(data.values()))

    # Filter data to only include arrays with more than 2 rows
    filtered_data = [item for item in all_data if item.shape[0] > 2]
    
    # Update all_data with the filtered data
    all_data = filtered_data

    # process data
    random.shuffle(all_data)
    split_idx = int(len(all_data) * split_ratio)
    
    # split data
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # save data
    torch.save(train_data, train_dest)
    torch.save(val_data, val_dest)
    
def get_dataloader(
        data_paths, 
        format_config, 
        mode,
        use_transposition, 
        neg_enhance,
        batch_size, 
        num_workers,
        use_sequence_collate_fn=False,
        shuffle=True
        ) -> DataLoader:
    
    dataset = OneBarChunkDataset(
        data_paths, 
        format_config, 
        mode=mode,
        use_transposition=use_transposition, 
        use_negative_enhance=neg_enhance
        )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle,
        collate_fn=sequence_collate_fn if use_sequence_collate_fn else torch.utils.data.dataloader.default_collate
        )

if __name__ == '__main__':
    p1 = r"C:\Users\cunn2\OneDrive\DSML\Project\thesis-repo\data\exp1\maestro_one_bar_segments_nr.pt"
    p2 = r"C:\Users\cunn2\OneDrive\DSML\Project\thesis-repo\data\exp1\mtc_one_bar_segments_nr.pt"
    train_dest = r"C:\Users\cunn2\OneDrive\DSML\Project\thesis-repo\data\exp1\train_data.pt"
    val_dest = r"C:\Users\cunn2\OneDrive\DSML\Project\thesis-repo\data\exp1\val_data.pt"
    produce_train_test_data(
        data_paths=[p1, p2],
        train_dest=train_dest,
        val_dest=val_dest,
        split_ratio=0.8
        )
