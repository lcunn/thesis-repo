from typing import List, Dict
from abc import ABC, abstractmethod

import magenta.music as mm
from note_seq.midi_io import note_sequence_to_midi_file
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np
import os
import tensorflow.compat.v1 as tf

from plagdet.src.defaults import *
from plagdet.src.embeddings.music_vae.extract_melody_custom import extract_melodies_custom
from plagdet.src.embeddings.music_vae.config import Config, MEL_2BAR_CUSTOM_CONFIG

tf.disable_v2_behavior()

# Necessary until pyfluidsynth is updated (>1.2.5).
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Mel2barEncoder():

    def __init__(
            self,
            use_custom_config: bool = False,
            custom_config: Config = MEL_2BAR_CUSTOM_CONFIG,
            model_path: str = MUSICVAE_2BAR_BIG_PATH
        ):
        if use_custom_config:
            self.config = custom_config
        else:
            self.config = configs.CONFIG_MAP[MUSICVAE_2BAR_BIG_CONFIG]
        self.model = TrainedModel(
            self.config, 
            batch_size=4,
            checkpoint_dir_or_path=model_path
        )

    @abstractmethod
    def extract_melodies_from_midi(self, midi_path: str) -> List[mm.NoteSequence]:
        pass

    def extract_melodies_from_midi_with_config(self, midi_path: str) -> List[mm.NoteSequence]:
        with open(midi_path, 'rb') as midi_file:
            midi_data = midi_file.read()
            input_seq = mm.midi_to_sequence_proto(midi_data)
        extracted_mels = self.config.data_converter.from_tensors(
            self.config.data_converter.to_tensors(input_seq)[1])
        return extracted_mels
    
    def encode_melody(self, melody: mm.NoteSequence) -> np.ndarray:
        extracted_tensors = self.model._config.data_converter.to_tensors(melody)
        if not extracted_tensors.inputs:
            # raise Exception('No examples extracted from NoteSequence: %s' % melody)
            return []
        # we might want to just take the first value here instead of looping
        vectors = [
            self.model.encode_tensors(
                [extracted_tensors.inputs[i]],
                [extracted_tensors.lengths[i]],
                [extracted_tensors.controls[i]]
                )[0]
            for i in range(len(extracted_tensors.inputs))
        ]
        return vectors
    
    def encode_midis(
            self, 
            midi_paths: List[str],
        ) -> Dict[str, List[np.ndarray]]:

        vectors = {}
        for path in midi_paths:
            extracted_mels = self.extract_melodies_from_midi(path)
            vectors[path] = []
            for mel in extracted_mels:
                vectors[path].extend(self.encode_melody(mel))
        return vectors

class Mel2barEncoderDefault(Mel2barEncoder):

    def extract_melodies_from_midi(self, midi_path: str) -> List[mm.NoteSequence]:
        with open(midi_path, 'rb') as midi_file:
            midi_data = midi_file.read()
            input_seq = mm.midi_to_sequence_proto(midi_data)
        extracted_mels = self.config.data_converter.from_tensors(
            self.config.data_converter.to_tensors(input_seq)[1])
        return extracted_mels

class Mel2barEncoderCustom(Mel2barEncoder):

    def extract_melodies_from_midi(self, midi_path: str) -> List[mm.NoteSequence]:
        mel_dict = extract_melodies_custom(midi_path)
        return [mel for melodies in mel_dict.values() for mel in melodies]
