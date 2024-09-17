import logging
import os
import numpy as np
import pandas as pd
from mido import MidiFile

from typing import Optional, Union, List, Tuple, Any, Dict

from sms.src.synthetic_data_mido.utils import calculate_total_bars, get_midi_info, calculate_bar_duration
from sms.src.synthetic_data_mido.mido_midi_inserter import MidiInserter
from sms.src.synthetic_data_mido.mido_midi_modifier import MidiModifier
from sms.src.log import configure_logging
from sms.defaults import *

configure_logging()
logger = logging.getLogger(__name__)

EXTRACT_BAR_LENGTH = 2

class PairGenerator:

    def __init__(self) -> None:
        pass

    def generate_pair(
            self, 
            original_path: str, 
            target_path: str
            ) -> Tuple[MidiFile, MidiFile, Dict[str, Any]]:
        """
        Takes an original song and synthesises a song which plagiarises it in some way.
        Plagiarism here meaning taking a segment, manipulating it, and inserting it into the target song.
        Returns the original and plagiarised MIDI files and the metadata describing the plagiarism.
        """
        mod = MidiModifier()
        ins = MidiInserter()

        original_midi = MidiFile(original_path)
        original_midi_info = get_midi_info(original_path)
        original_bar_duration = calculate_bar_duration(*original_midi_info)
        logger.info(f'Original bar duration: {original_bar_duration}')

        target_midi_info = get_midi_info(target_path)
        target_bar_duration = calculate_bar_duration(*target_midi_info)
        logger.info(f'Target bar duration: {target_bar_duration}')

        # randomly select 2 bars from the original song
        num_bars_in_original = calculate_total_bars(original_path)
        segment_starting_bar = np.random.randint(1, np.floor(num_bars_in_original-EXTRACT_BAR_LENGTH))
        segment_starting_time_secs = segment_starting_bar * original_bar_duration
        segment_path = 'segment.mid'
        ins.extract_bars(original_path, segment_path, segment_starting_bar, EXTRACT_BAR_LENGTH)

        # modify the segment
        modified_segment_path = 'modified_segment.mid'
        mod.set_midi(segment_path)
        mod.generate_and_set_midi_modifier_config(
            use_shift_entire_midi_pitch = True,
            use_shift_selected_notes_pitch = True,
            use_change_note_durations = True,
            use_delete_notes = True,
        )
        mod.modify_midi(output_path=modified_segment_path)

        # randomly insert the modified segment into the target song
        modified_target_path = 'modified_target.mid'
        num_bars_in_target = calculate_total_bars(target_path)
        segment_insertion_starting_bar = np.random.randint(1, np.floor(num_bars_in_target-EXTRACT_BAR_LENGTH))
        segment_insertion_starting_time_secs = segment_insertion_starting_bar * target_bar_duration
        ins.insert_segment(modified_segment_path, target_path, segment_insertion_starting_bar, modified_target_path)

        metadata = {
            'segment_starting_bar': segment_starting_bar,
            'segment_starting_time_secs': segment_starting_time_secs,
            'segment_insertion_starting_bar': segment_insertion_starting_bar,
            'segment_insertion_starting_time_secs': segment_insertion_starting_time_secs,
            'midi_modifier_config': mod.config
        }

        original_midi = MidiFile(original_path)
        plagiarised_midi = MidiFile(modified_target_path)

        return original_midi, plagiarised_midi, metadata

    def convert_directory_to_pairs(self, original_dir: str) -> None:
        """
        original_dir: path to a directory of monophonic MIDI files
        """

        # get all midis in the directory

        midis = []
        for root, dirs, files in os.walk(original_dir):
            for file in files:
                if file.endswith('.mid'):
                    midis.append(os.path.join(root, file))

        # split into pairs

        midis = midis[:int(2*np.floor(len(midis)/2))]
        originals = midis[::2]
        targets = midis[1::2]

        # open tracking files

        if not os.path.exists(SYNTHETIC_PAIR_TRACKER_PATH):
            pair_tracker_df = pd.DataFrame(
                columns=[
                'pair_id',
                'original_midi_path',
                'plagiarised_midi_path'
                ])
        else:
            pair_tracker_df = pd.read_csv(SYNTHETIC_PAIR_TRACKER_PATH)

        if not os.path.exists(SYNTHETIC_PAIR_METADATA_PATH):
            metadata_df = pd.DataFrame(
                columns=[
                'pair_id', 
                'midi_modifier_config', 
                'segment_starting_bar',
                'segment_starting_time_secs',
                'segment_insertion_starting_bar',
                'segment_insertion_starting_time_secs'
                ])
        else:
            metadata_df = pd.read_pickle(SYNTHETIC_PAIR_METADATA_PATH)

        # generate pairs
        current_pair_id = len(pair_tracker_df)

        for original, target in zip(originals, targets):
            logger.info(f'Generating pair {current_pair_id}/{len(originals)}')

            try:
                current_pair_id += 1
                original_midi, plagiarised_midi, metadata = self.generate_pair(original, target)

                original_path = os.path.join(ORIGINAL_SYNTHETIC_MIDIS_PATH, f'original_{current_pair_id}.mid')
                plagiarised_path = os.path.join(PLAGIARISED_SYNTHETIC_MIDIS_PATH, f'plagiarised_{current_pair_id}.mid')

                original_midi.save(original_path)
                plagiarised_midi.save(plagiarised_path)

                # write to pair tracker dataframe
                pair_tracker_df.loc[current_pair_id] = [
                    current_pair_id,
                    original_path,
                    plagiarised_path
                ]

                # write to metadata dataframe
                metadata_df.loc[current_pair_id] = [
                    current_pair_id,
                    metadata['midi_modifier_config'],
                    metadata['segment_starting_bar'],
                    metadata['segment_starting_time_secs'],
                    metadata['segment_insertion_starting_bar'],
                    metadata['segment_insertion_starting_time_secs']
                ]

                logger.info(f'Saved pair {current_pair_id}/{len(originals)}')

            except Exception as e:
                logger.error(f'Error generating pair {current_pair_id}: {str(e)}')
                continue
        
        pair_tracker_df.to_csv(SYNTHETIC_PAIR_TRACKER_PATH, index=False)
        metadata_df.to_pickle(SYNTHETIC_PAIR_METADATA_PATH)

def turn_path_into_pairs(path: str = MONOPHONIC_MIDIS_PATH) -> None:
    pair_generator = PairGenerator()
    pair_generator.convert_directory_to_pairs(path)

if __name__ == '__main__':
    turn_path_into_pairs()

    



