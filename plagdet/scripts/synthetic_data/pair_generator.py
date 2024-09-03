import logging
import numpy as np
import pretty_midi
import mido
from mido import MidiFile, MidiTrack, MetaMessage

from typing import Optional, Union

from plagdet.scripts.synthetic_data.utils import calculate_total_bars, get_tempo_and_time_signature, set_midi_tempo
from plagdet.src.utils.log import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:

    def generate_pair(
        self,
        original_track_path: str,
        target_track_path: str,
    ):
        pass
        
    def extract_bars(self, midi_path: str, dest_path: str, start_bar: int, num_bars: int = 2):
        # Load the MIDI file
        mid = MidiFile(midi_path)
        
        # Calculate tempo and time signature
        tempo, time_signature = get_tempo_and_time_signature(midi_path)
        logger.info(f'Original file: Tempo: {tempo}, Time Signature: {time_signature}, Ticks per beat: {mid.ticks_per_beat}')

        ticks_per_beat = mid.ticks_per_beat

        # Calculate ticks per bar based on the time signature
        beats_per_bar = time_signature[0]
        ticks_per_bar = beats_per_bar * ticks_per_beat
        
        # Calculate start tick and end tick
        start_tick = (start_bar - 1) * ticks_per_bar
        end_tick = start_tick + num_bars * ticks_per_bar
        
        # Create a new MIDI file
        new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)

        # Add tempo and time signature information
        tempo_in_microseconds = mido.bpm2tempo(tempo)
        new_track.append(MetaMessage('set_tempo', tempo=tempo_in_microseconds, time=0))
        new_track.append(MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1], clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

        current_tick = 0    
        for track in mid.tracks:
            for msg in track:
                current_tick += msg.time
                if start_tick <= current_tick < end_tick:
                    # Adjust message time if it's the first message
                    if current_tick == start_tick:
                        msg.time = 0
                    new_track.append(msg)
                elif current_tick >= end_tick:
                    break

        # Save the new MIDI file
        new_mid.save(dest_path)
        set_midi_tempo(dest_path, tempo)

        # # Log information about the new file
        # new_mid_check = MidiFile(dest_path)
        # new_tempo, new_time_signature = get_tempo_and_time_signature(dest_path)
        # logger.info(f'New file: Tempo: {new_tempo}, Time Signature: {new_time_signature}, Ticks per beat: {new_mid_check.ticks_per_beat}')

        # # Compare original and new file
        # logger.info(f'Tempo difference: {tempo - new_tempo}')
        # logger.info(f'Time signature difference: {time_signature != new_time_signature}')
        # logger.info(f'Ticks per beat difference: {mid.ticks_per_beat - new_mid_check.ticks_per_beat}')

        
    def apply_transformation(self, original_bar: np.ndarray, target_bar: np.ndarray):
        pass
