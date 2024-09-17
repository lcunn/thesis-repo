import logging
import mido
import os
import numpy as np
from dataclasses import dataclass
from mido import MidiFile, MidiTrack, MetaMessage

from typing import Optional, Union, List, Tuple

from sms.src.synthetic_data_mido.utils import calculate_total_bars, get_midi_info, set_midi_tempo, calculate_bar_duration
from sms.src.log import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class MidiInserter:

    # ----------------------------------------------------------------------------------------
    # pre-processing
    # ----------------------------------------------------------------------------------------

    def condense_midi_to_one_track(self, input_path: str, output_path: str) -> None:
        # Load the MIDI file
        mid = MidiFile(input_path)
        
        # Create a new MIDI file and a single track
        new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)
        
        # List to hold all messages with their absolute times
        all_messages = []
        
        # Iterate through all tracks and collect messages
        for track in mid.tracks:
            current_tick = 0
            for msg in track:
                current_tick += msg.time
                all_messages.append((current_tick, msg))
        
        # Sort messages by their absolute time
        all_messages.sort(key=lambda x: x[0])
        
        # Add messages to the new track, adjusting the time to be relative
        last_tick = 0
        for abs_tick, msg in all_messages:
            relative_time = abs_tick - last_tick
            new_msg = msg.copy(time=relative_time)
            new_track.append(new_msg)
            last_tick = abs_tick
        
        # Save the new MIDI file
        new_mid.save(output_path)

    def scale_notes(self, midi1_path: str, midi2_path: str, output_path: str):
        """
        Scales notes from midi1 so that the bar duration is the same as midi2, under the tempo and ticks of midi2.
        """
        # Get info for both MIDI files
        tempo1, time_sig1, tpb1 = get_midi_info(midi1_path)
        tempo2, time_sig2, tpb2 = get_midi_info(midi2_path)
        
        # Calculate bar durations
        bar_duration1 = calculate_bar_duration(tempo2, time_sig2, tpb1)
        bar_duration2 = calculate_bar_duration(tempo2, time_sig2, tpb2)
        
        # Calculate scaling factor
        scaling_factor = bar_duration2 / bar_duration1
        
        # Open MIDI files
        midi1 = MidiFile(midi1_path)
        midi2 = MidiFile(midi2_path)
        
        # Create new MIDI file with MIDI 2's properties
        new_midi = MidiFile(ticks_per_beat=tpb2)
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)
        
        # Add tempo and time signature from MIDI 2
        new_track.append(MetaMessage('set_tempo', tempo=tempo2, time=0))
        new_track.append(MetaMessage('time_signature', numerator=time_sig2[0], denominator=time_sig2[1], clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        
        # Transfer and scale note-on messages from MIDI 1
        for track in midi1.tracks:
            for msg in track:
                if msg.type == 'note_on':
                    new_msg = msg.copy()
                    new_msg.time = int(msg.time * scaling_factor)
                    new_track.append(new_msg)
        
        # Save the new MIDI file
        new_midi.save(output_path)

    # ----------------------------------------------------------------------------------------
    # extract and insert segments
    # ----------------------------------------------------------------------------------------
     

    def extract_bars(self, midi_path: str, dest_path: str, start_bar: int, num_bars: int = 2):

        total_bars_in_song = calculate_total_bars(midi_path)
        assert start_bar >= 1, f'Start bar must be at least 1. Got {start_bar}'
        assert start_bar + num_bars <= total_bars_in_song-1, f'Start bar + num bars must be less than total bars in song - 1. Got {start_bar} + {num_bars} = {start_bar + num_bars} > {total_bars_in_song-1}'
        # Load the MIDI file
        one_track_path = f'one_track.mid'
        self.condense_midi_to_one_track(midi_path, one_track_path)
        mid = MidiFile(one_track_path)
        
        # Calculate tempo and time signature
        tempo, time_signature, ticks_per_beat = get_midi_info(midi_path)
        bpm = mido.tempo2bpm(tempo)
        logger.info(f'Original file: BPM: {bpm}, Time Signature: {time_signature}, Ticks per beat: {ticks_per_beat}')

        # Calculate ticks per bar based on the time signature
        beats_per_bar = time_signature[0]
        ticks_per_bar = beats_per_bar * ticks_per_beat
        
        # Calculate start tick and end tick
        start_tick = (start_bar) * ticks_per_bar
        end_tick = start_tick + num_bars * ticks_per_bar
        
        # Create a new MIDI file
        new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)

        # Add tempo and time signature information
        tempo_in_microseconds = mido.bpm2tempo(bpm)
        new_track.append(MetaMessage('set_tempo', tempo=tempo_in_microseconds, time=0))
        new_track.append(MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1], clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

        current_tick = 0    
        for msg in mid.tracks[0]:
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

    def insert_segment(self, segment_path: str, target_path: str, start_bar: int, dest_path: str) -> None:

        target_one_track_path = f'target_one_track.mid'
        segment_one_track_path = f'segment_one_track.mid'

        self.condense_midi_to_one_track(segment_path, segment_one_track_path)
        self.condense_midi_to_one_track(target_path, target_one_track_path)
        
        target_mid = MidiFile(target_one_track_path)

        tempo, time_signature, tpb_target = get_midi_info(target_one_track_path)
        
        # Scale the segment notes to match the target tempo and ticks per beat
        scaled_segment_path = 'temp_scaled_segment.mid'
        self.scale_notes(segment_one_track_path, target_one_track_path, scaled_segment_path)
        segment_mid = MidiFile(scaled_segment_path)
        
        num_bars_in_segment = calculate_total_bars(scaled_segment_path)
        ticks_per_bar = time_signature[0] * tpb_target
        start_tick = start_bar * ticks_per_bar
        end_tick = start_tick + int(num_bars_in_segment * ticks_per_bar)
        
        new_mid = MidiFile(ticks_per_beat=tpb_target)
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)
        
        # Add tempo and time signature information
        new_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        new_track.append(MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1], clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

        pre_segment_msgs = []
        post_segment_msgs = []
        current_tick = 0
        for msg in target_mid.tracks[0]:
            current_tick += msg.time
            if current_tick < start_tick:   
                pre_segment_msgs.append(msg)
            elif current_tick > end_tick:
                post_segment_msgs.append(msg)

        segment_msgs = []
        for msg in segment_mid.tracks[0]:
            if msg.type == 'note_on':
                segment_msgs.append(msg)

        def ensure_all_notes_off(messages):
            active_notes = set()
            for msg in messages:
                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        active_notes.add(msg.note)
                    else:
                        active_notes.discard(msg.note)
        
            note_off_messages = []
            for note in active_notes:
                note_off_messages.append(mido.Message('note_on', note=note, velocity=0, time=0))
            
            return messages + note_off_messages
        
        pre_segment_msgs = ensure_all_notes_off(pre_segment_msgs)
        segment_msgs = ensure_all_notes_off(segment_msgs)

        new_track.extend(pre_segment_msgs)
        new_track.extend(segment_msgs)
        new_track.extend(post_segment_msgs)
        
        new_mid.save(dest_path)
        os.remove(scaled_segment_path)
