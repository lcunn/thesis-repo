import logging
import mido
import os
import numpy as np
from dataclasses import dataclass
from mido import MidiFile, MidiTrack, MetaMessage

from typing import Optional, Union, List, Tuple

from sms.src.log import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

@dataclass
class Note:
    start_idx: int
    end_idx: int

@dataclass
class MidiModifierConfig:
    use_transposition: bool = False
    transposition_semitone: Optional[int] = None

    use_shift_selected_notes_pitch: bool = False
    selected_notes_pitch_shifts: Optional[List[Tuple[Note, int]]] = None

    use_change_note_durations: bool = False
    note_scale_factors: Optional[List[Tuple[Note, float]]] = None

    use_delete_notes: bool = False
    notes_to_delete: Optional[List[Note]] = None

TRANSPOSITION_SEMITONE_RANGE = (-16, 16)

NOTES_TO_PITCH_SHIFT_PERCENTAGE = 0.2
NOTE_PITCH_SHIFT_RANGE = (-12, 12)

NOTES_TO_SCALE_PERCENTAGE = 0.1
NOTE_DURATION_SCALE_RANGE = (1, 3)

NOTES_TO_DELETE_PERCENTAGE = 0.1

class MidiModifier:
    """
    Class to apply plagiarism-esque transformations to a single-tracked MIDI file.
    """

    def __init__(self):
        self.midi_path = None
        self.midi = None

    def set_midi(self, midi_path: str):
        self.midi_path = midi_path
        self.midi = MidiFile(midi_path)

    def generate_and_set_midi_modifier_config(
            self,
            use_transposition: bool = False,
            use_shift_selected_notes_pitch: bool = False,
            use_change_note_durations: bool = False,
            use_delete_notes: bool = False,
        ) -> None:
        """
        Randomly picks the shifts and indices for the config, and sets self.config.
        """

        notes_in_midi = self.identify_note_events(self.midi)
        num_notes = len(notes_in_midi)
        config = MidiModifierConfig()

        if use_transposition:
            config.use_transposition = True
            # randomly select to shift semitones by max 24 semitones up or down
            config.transposition_semitone = int(np.random.randint(TRANSPOSITION_SEMITONE_RANGE[0], TRANSPOSITION_SEMITONE_RANGE[1]))
            logger.info(f'Transposing entire MIDI by {config.transposition_semitone} semitones')
        
        if use_shift_selected_notes_pitch:
            config.use_shift_selected_notes_pitch = True
            # randomly select to x% of the notes to shift
            num_shifts = np.random.randint(1, max(2, np.floor(num_notes*NOTES_TO_PITCH_SHIFT_PERCENTAGE)))

            logger.info(f'Pitch shifting {num_shifts} notes')
            # randomly select which notes to shift
            shift_idxs = np.random.choice(num_notes, num_shifts, replace=False)
            # randomly select to shift semitones by up to 12 semitones up or down
            shifts = np.random.randint(NOTE_PITCH_SHIFT_RANGE[0], NOTE_PITCH_SHIFT_RANGE[1], num_shifts)
            logger.info(f'Pitch shifts: {shifts}')

            config.selected_notes_pitch_shifts = []
            for shift_idx, shift in zip(shift_idxs, shifts):
                config.selected_notes_pitch_shifts.append((notes_in_midi[shift_idx], int(shift)))

        if use_change_note_durations:
            config.use_change_note_durations = True
            # randomly select to x% of the notes to shift
            num_shifts = np.random.randint(1, max(2, np.floor(num_notes*NOTES_TO_SCALE_PERCENTAGE)))
            logger.info(f'Scaling {num_shifts} notes')
            # randomly select which notes to shift
            shift_idxs = np.random.choice(num_notes, num_shifts, replace=False)
            # randomly select to shift semitones by up to 12 semitones up or down
            shifts = np.random.uniform(NOTE_DURATION_SCALE_RANGE[0], NOTE_DURATION_SCALE_RANGE[1], num_shifts)
            logger.info(f'Scaling factors: {shifts}')

            config.note_scale_factors = []
            for shift_idx, shift in zip(shift_idxs, shifts):
                config.note_scale_factors.append((notes_in_midi[shift_idx], shift))

        if use_delete_notes:
            config.use_delete_notes = True
            # randomly select to x% of the notes to delete
            num_deletes = np.random.randint(1, max(2, np.floor(num_notes*NOTES_TO_DELETE_PERCENTAGE)))
            logger.info(f'Deleting {num_deletes} notes')
            # randomly select which notes to delete
            delete_idxs = np.random.choice(num_notes, num_deletes, replace=False)
            logger.info(f'Delete indices: {delete_idxs}')

            config.notes_to_delete = []
            for delete_idx in delete_idxs:
                config.notes_to_delete.append(notes_in_midi[delete_idx])

        self.config = config

    def modify_midi(
            self, 
            midi: Optional[MidiFile] = None, 
            config: Optional[MidiModifierConfig] = None,
            output_path: Optional[str] = None
        ) -> Optional[MidiFile]:
        """
        Applies the modifications described in the config to the MIDI file.
        If output_path, it saves the modified MIDI file to the path. Else, it returns the modified MidiFile.
        """

        midi = midi or self.midi
        config = config or self.config
        
        if config.use_transposition:
            midi = self.transpose_midi(midi, config.transposition_semitone)

        if config.use_shift_selected_notes_pitch:
            midi = self.shift_selected_notes_pitch(midi, config.selected_notes_pitch_shifts)

        if config.use_change_note_durations:
            midi = self.change_note_durations(midi, config.note_scale_factors)

        if config.use_delete_notes:
            midi = self.delete_notes(midi, config.notes_to_delete)

        if output_path:
            midi.save(output_path)
        else:
            return midi
        
    # ----------------------------------------------------------------------------------------
    # util
    # ---------------------------------------------------------------------------------------- 
    @staticmethod
    def identify_note_events(midi: MidiFile) -> List[Note]:
        """
        Given a MidiFile, obtain a list of pairs (note_on_idx, note_off_idx) describing notes played in the MidiFile.
        idx refers to the index inside the list of all messages in the MidiFile.
        """
        note_events = []
        active_notes = {}
        for idx, msg in enumerate(midi.tracks[0]):
            if msg.type == 'note_on':
                if msg.velocity > 0:
                    active_notes[msg.note] = idx
                else:
                    if msg.note in active_notes:
                        note_events.append((active_notes[msg.note], idx))
                        del active_notes[msg.note]
            elif msg.type == 'note_off':
                if msg.note in active_notes:
                    note_events.append((active_notes[msg.note], idx))
                    del active_notes[msg.note]
        
        return note_events

    # ----------------------------------------------------------------------------------------
    # modification methods
    # ---------------------------------------------------------------------------------------- 
    @staticmethod
    def transpose_midi(midi: MidiFile, semitone_shift: int) -> MidiFile:
        """
        Shifts the pitch of the entire MIDI file by a given number of semitones.
        """
        new_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        for msg in midi.tracks[0]:
            if msg.type == 'note_on':
                msg.note += semitone_shift
            new_track.append(msg)
        
        return new_midi

    @staticmethod
    def shift_selected_notes_pitch(midi: MidiFile, notes_to_change: List[Tuple[Note, int]]) -> MidiFile:
        """
        Applies the shift described by each ((note_on_idx, note_off_idx), shift) tuple to the MIDI file.
        """
        new_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        note_on_dict = {note_on_idx:shift for ((note_on_idx, note_off_idx), shift) in notes_to_change}
        note_off_dict = {note_off_idx:shift for ((note_on_idx, note_off_idx), shift) in notes_to_change}

        for idx, msg in enumerate(midi.tracks[0]):
            new_msg = msg.copy()
            if idx in note_on_dict.keys():
                shift = note_on_dict[idx]
                new_msg.note = int(new_msg.note + shift)
            elif idx in note_off_dict.keys():
                shift = note_off_dict[idx]
                new_msg.note = int(new_msg.note + shift)
            new_track.append(new_msg)
        
        return new_midi

    @staticmethod
    def change_note_durations(midi: MidiFile, notes_to_scale: List[Tuple[Note, float]]) -> MidiFile:
        """
        Changes the duration of the notes described by each (Note, duration_multiplier) tuple.
        Note is (note_on_idx, note_off_idx) referring to indices in the track's message list.
        """
        new_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        note_scale_dict = {note_on_idx+1: scale for (note_on_idx, note_off_idx), scale in notes_to_scale}

        for idx, msg in enumerate(midi.tracks[0]):
            new_msg = msg.copy()
            if idx in note_scale_dict.keys():
                new_msg.time = int(new_msg.time * note_scale_dict[idx])
            new_track.append(new_msg)

        return new_midi
    
    @staticmethod
    def change_note_durations(midi: MidiFile, notes_to_scale: List[Tuple[Note, float]]) -> MidiFile:
        """
        Changes the duration of the notes described by each (Note, duration_multiplier) tuple.
        Note is (note_on_idx, note_off_idx) referring to indices in the track's message list.
        """
        new_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        note_on_dict = {note_on_idx: (note_off_idx, scale) for (note_on_idx, note_off_idx), scale in notes_to_scale}
        cumulative_time = 0
        time_adjustments = {}

        # calculate time adjustments for each note
        for idx, msg in enumerate(midi.tracks[0]):
            cumulative_time += msg.time
            if idx in note_on_dict.keys():
                # This is a note_on message that needs to be scaled
                start = idx
                end, scale = note_on_dict[idx]
                original_duration = sum(m.time for m in midi.tracks[0][start+1:end+1])
                new_duration = int(original_duration * scale)
                # we need to adjust the time of note. so we increase the delta of the next note
                time_adjustments[start+1] = new_duration - original_duration

        # apply time adjustments    
        for idx, msg in enumerate(midi.tracks[0]):
            new_msg = msg.copy()
            if idx in time_adjustments.keys():
                new_msg.time += int(time_adjustments[idx])
            new_track.append(new_msg)

        return new_midi

    @staticmethod
    def delete_notes(midi: MidiFile, notes_to_delete: List[Note]) -> MidiFile:
        """
        Deletes the notes described by each (note_on_idx, note_off_idx) tuple.
        """
        new_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        note_on_list, note_off_list = zip(*notes_to_delete)
        note_on_list, note_off_list = list(note_on_list), list(note_off_list)

        for idx, msg in enumerate(midi.tracks[0]):
            if idx not in note_on_list and idx not in note_off_list:
                new_track.append(msg)
        
        return new_midi