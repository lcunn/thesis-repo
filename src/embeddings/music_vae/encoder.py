from typing import Dict, List, Optional

from src.midi.midi_processor import MidiProcessor

from src.embeddings.music_vae.split_ns import split_into_2_bar_segments, extract_2_bar_melodies

# from note_seq.protobuf.music_pb2 import NoteSequence
from note_seq.sequences_lib import extract_subsequence
from note_seq.midi_io import midi_file_to_note_sequence, note_sequence_to_midi_file
from magenta.music import sequences_lib
from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline, Splitter, Quantizer

"""
Using the encoder from the notebook...



"""

class MusicVAEncoder(MidiProcessor):

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._note_sequence = midi_file_to_note_sequence(file_path)
        self._melodies = None

    def extract_melodies(
        self,
        quantize_steps_per_quarter: int = 8
    ) -> None:
        """
        Extracts all melodies from the list of 2 bar segments using the magenta MelodyExtractor and stores in instance.

        Args:
            quantize_steps_per_quarter: whether to quantize the melody.
        """

        two_bar_segments = split_into_2_bar_segments(
            self._note_sequence, 
            quantize = True, 
            quantize_steps_per_quarter = quantize_steps_per_quarter)
        
        melodies = {}

        for start_bar, segment in enumerate(two_bar_segments):
            mels = extract_2_bar_melodies(segment)
            if mels:
                melodies[str(start_bar)] = mels
            else:
                melodies[str(start_bar)] = None

        self._melodies = melodies

        @property
        def melodies(self):
            return self._melodies
            
        



        