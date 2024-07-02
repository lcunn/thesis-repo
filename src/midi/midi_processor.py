from typing import str

import mido

class MidiProcessor:
    """
    Base class for midi processing.
    """
    def __init__(self, file_path):
        self._file_path = file_path
        self._midi_file = self.load_midi()

    @property
    def midi_file(self):
        return self._midi_file
    
    def load_midi(self):
        try:
            midi_file = mido.MidiFile(self._file_path)
            print(f"Loaded MIDI file: {self._file_path}")
            return midi_file
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file: {e}")