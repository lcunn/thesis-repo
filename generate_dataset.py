import logging
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import os
import numpy as np
import mido
from mido import MidiFile, MidiTrack, MetaMessage
import torch
from torch.utils.data import Dataset

from sms.src.log import configure_logging
from sms.src.synthetic_data.midi_inserter import MidiInserter
from sms.src.synthetic_data.midi_modifier import MidiModifier, MidiModifierConfig
from sms.src.synthetic_data.pair_generator import SegmentPerturber
from sms.src.embeddings.music_vae.two_bar_encoder import Mel2barEncoderCustom

class MusicVAEMelodyPairGenerator:
    def __init__(self):
        self.extractor = MidiInserter()
        self.modifier = MidiModifier()
        self.perturber = SegmentPerturber()
        self.model = Mel2barEncoderCustom()

    def generate_pair(self, midi_path: str, output_path: str) -> None:
        """
        Takes a MIDI file and applies the segment perturber to it.
        """
        perturber_config = self.perturber(midi_path, output_path)
        
        embeddings = self.model.encode_midis([midi_path, output_path])
        original_embedding = embeddings[midi_path]
        perturbed_embedding = embeddings[output_path]
        
        #TODO: what to return?
    
    def process_directory(self, directory_path, output_path):
        pairs = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.mid'):
                    midi_path = os.path.join(root, file)
                    pair = self.generate_pair(midi_path)
                    pairs.append(pair)
        
        # Save pairs and embeddings
        torch.save(pairs, output_path)

class MelodyPairDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'original_embedding': item['original_embedding'],
            'perturbed_embedding': item['perturbed_embedding']
        }

def generate_dataset(input_dir, output_path, model_path):
    generator = MelodyPairGenerator(model_path)
    generator.process_directory(input_dir, output_path)

if __name__ == '__main__':
    input_dir = 'path/to/monophonic/midis'
    output_path = 'path/to/processed/data.pt'
    model_path = 'path/to/musicvae/model'
    generate_dataset(input_dir, output_path, model_path)