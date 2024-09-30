import numpy as np
from typing import List, Optional

class InputFormatter:
    """
    The raw segment dataset has note_arrays where each note has values [duration, pitch].
    """
    def __init__(
        self,
        normalize_octave: bool = False,
        make_relative_pitch: bool = False,
        quantize: bool = False,
        piano_roll: bool = False,
        steps_per_bar: int = 32,
        bars: int = 1,
        rest_pitch: int = -1,
        pad_sequence: bool = False,
        pad_val: int = -1000,
        goal_seq_len: int = 12
    ):
        """
        The first three can all be used simultaneously.
        If piano_roll is True, then then rest are forced to False.
        """
        # removes possiblity of errors
        self.bars = bars
        self.steps_per_bar = steps_per_bar
        self.rest_pitch = rest_pitch
        self.pad_sequence = pad_sequence if not (quantize or piano_roll) else False # if quantize or piano_roll, we don't pad
        self.pad_val = pad_val
        self.goal_seq_len = goal_seq_len
        if piano_roll:
            self.config_piano_roll = True
            self.config_normalize_octave = False
            self.config_make_relative_pitch = False
            self.config_quantize = False
        else:
            self.config_piano_roll = False
            self.config_normalize_octave = normalize_octave
            self.config_make_relative_pitch = make_relative_pitch
            self.config_quantize = quantize

    def __call__(self, note_array: np.ndarray) -> np.ndarray:
        """
        note_array in form [duration, pitch].
        a one bar segment will be 4 beats.
        applies formatting to the note_array.
        """
        note_array = np.copy(note_array)
        if self.config_piano_roll:
            note_array = self.make_piano_roll(note_array)
        if self.config_normalize_octave:
            note_array = self.normalize_octave(note_array)
        if self.config_make_relative_pitch:
            note_array = self.make_relative_pitch(note_array)
        if self.config_quantize:
            note_array = self.quantize(note_array)
        if self.pad_sequence:
            note_array = self.pad_sequence_array(note_array)
        return note_array
    
    def pad_sequence_array(self, note_array: np.ndarray) -> np.ndarray:
        if len(note_array) >= self.goal_seq_len:
            return note_array[:self.goal_seq_len]
        else:
            pad_length = self.goal_seq_len - len(note_array)
            padding = np.zeros((pad_length, 2))
            padding[:, :] = self.pad_val
            return np.vstack((note_array, padding))

    def normalize_octave(self, note_array: np.ndarray) -> np.ndarray:
        """
        note_array in form [duration, pitch].
        normalize the range of the pitch to 12-24, with 0 for rests.
        returns a note_array in form [duration, pitch_normalized_to_octave].
        """
        normalized_note_array = np.copy(note_array)
        non_rest_mask = normalized_note_array[:, 1] != self.rest_pitch
        normalized_note_array[non_rest_mask, 1] = (normalized_note_array[non_rest_mask, 1] % 12) + 12
        return normalized_note_array

    def make_relative_pitch(self, note_array: np.ndarray) -> np.ndarray:
        """
        note_array in form [duration, pitch].
        make the pitch relative to the previous note.
        returns a note_array in form [duration, pitch_relative_to_previous_note].
        """
        relative_pitch_note_array = np.copy(note_array)
        for i in reversed(range(len(relative_pitch_note_array)-1)):
            idx = i+1
            relative_pitch_note_array[idx][1] = relative_pitch_note_array[idx][1] - relative_pitch_note_array[idx-1][1]
        return relative_pitch_note_array

    def quantize(self, note_array: np.ndarray) -> np.ndarray:
        """
        note_array where notes are in form [duration, pitch].
        transforms by dividing each bar into steps_per_bar bins. the value of each bin is the pitch of the note playing at the start.
        returns a (steps_per_bar * num_bars) length array.
        """
        steps_per_bar = self.steps_per_bar
        length = int(steps_per_bar * self.bars)

        # change the duration to 4 units per bar to steps_per_bar per bar
        note_array = np.copy(note_array)
        note_array[:, 0] *= steps_per_bar/4

        # add cumulative duration column
        cumulative_duration = np.cumsum(note_array[:, 0])
        note_array = np.column_stack((note_array, cumulative_duration))

        # for each duration step, pick the first note with a cumulative duration greater than the current step increment; 
        #this guarantees it was playing after the start of the increment
        quantized_note_array = np.array(
            [
                note_array[note_array[:, 2] > i][0][1] 
                for i in range(length)
            ]
        )
        
        return quantized_note_array
    
    def make_piano_roll(self, note_array: np.ndarray) -> np.ndarray:
        """
        Converts a note_array in form [duration, pitch] to a piano roll representation.
        
        Args:
        note_array (np.ndarray): Array of notes in [duration, pitch] format.
        steps_per_bar (int): Number of time steps per bar (default is 32).
        
        Returns:
        np.ndarray: A 2D numpy array, shape (128, steps_per_bar), representing the piano roll.
        """
        steps_per_bar = self.steps_per_bar

        quantized = self.quantize(note_array)
        
        piano_roll = np.zeros((128, steps_per_bar), dtype=int)
        
        for step, pitch in enumerate(quantized):
            if pitch != self.rest_pitch:  # not a rest
                pitch_idx = int(pitch) - 1  # Adjust for 1-127 range
                if 0 <= pitch_idx <= 127:
                    piano_roll[pitch_idx, step] = 1.0
        
        piano_roll = np.flipud(piano_roll)

        return piano_roll
