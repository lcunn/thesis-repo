from typing import Dict, List, Optional

from src.midi.midi_processor import MidiProcessor

from src.defaults.objects import NoteSequence
from src.embeddings.music_vae.split_ns import split_into_2_bar_segments, extract_2_bar_melodies

from note_seq.sequences_lib import extract_subsequence
from note_seq.midi_io import midi_file_to_note_sequence, note_sequence_to_midi_file
from magenta.music import sequences_lib
from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline, Splitter, Quantizer

from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np

"""
Using the encoder from the notebook...
"""

class MusicVAEncoder(MidiProcessor):

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._note_sequence = midi_file_to_note_sequence(file_path)
        self._two_bar_segments_q = split_into_2_bar_segments(self._note_sequence)
        self._two_bar_segments_unq = split_into_2_bar_segments(self._note_sequence, quantize=False)
        self._magenta_melodies = self._extract_magenta_melodies()

    def _extract_magenta_melodies(self) -> Dict[str, NoteSequence]:
        """
        Extracts all melodies from the list of 2 bar segments using the magenta MelodyExtractor and stores in a dictionary.
        Keys are a string representation of the starting bar of the 2 bar melody, e.g. '0'.
        """
        melodies = {}
        for start_bar, segment in enumerate(self._two_bar_segments_q):
            mels = extract_2_bar_melodies(segment)
            melodies[str(start_bar)] = mels
        return melodies

    def encode_melodies(self):
        """
        Encodes the extracted melodies using MusicVAE.
        Returns a dictionary with bar numbers as keys and encoded vectors as values.
        """
        config = configs.CONFIG_MAP['cat-mel_2bar_big']
        model = TrainedModel(
            config, batch_size=4,
            checkpoint_dir_or_path='models/music_vae/2_bar_melody/cat-mel_2bar_big')

        encoded_melodies = {}
        
        for start_bar, melodies in self._magenta_melodies.items():
            if melodies:  # Check if the list is not empty
                z, _, _ = model.encode(melodies)
                encoded_melodies[start_bar] = z.numpy()

        return encoded_melodies

    @property
    def magenta_melodies(self):
        return self._magenta_melodies
    
            
        



        