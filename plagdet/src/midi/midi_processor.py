# from typing import 

import mido
import pretty_midi

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

    def split_midi_by_instrument(self, output_prefix):
        # Load the MIDI file
        midi_data = pretty_midi.PrettyMIDI(self._file_path)
        
        # Iterate through each instrument
        for i, instrument in enumerate(midi_data.instruments):
            # Create a new PrettyMIDI object
            new_midi = pretty_midi.PrettyMIDI()
            # Add the instrument to the new MIDI
            new_midi.instruments.append(instrument)
            # Save the new MIDI file
            output_file = f"{output_prefix}_instrument_{i}.mid"
            new_midi.write(output_file)
            print(f"Saved {output_file}")
