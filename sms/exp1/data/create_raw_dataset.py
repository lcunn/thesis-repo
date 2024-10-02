import os
import torch
import logging
from typing import Dict

from sms.src.synthetic_data.midi_to_note_arrays import midi_to_note_array
from sms.src.log import configure_logging
from sms.defaults import MAESTRO_PATH, MTC_PATH, MAESTRO_SEGMENTS_PATH, MTC_SEGMENTS_PATH

logger = logging.getLogger(__name__)
configure_logging()

def monophonic_midis_to_note_arrays(
        input_output_dict: Dict[str, str],
        num_bars: int = 1, 
        start_bar_proportion: float = 0.4,
        remove_rests: bool = False
        ) -> None:
    """
    Takes a directories of monophonic MIDI files and converts them to note arrays.

    Args:
        input_output_dict (Dict[str, str]): Dictionary with keys as input paths and values as output paths.
        num_bars (int): Number of bars to extract.
        start_bar_proportion (float): Proportion of the way through the song to start.
        remove_rests (bool): Whether to remove rests from the note array. 'Removes' rests by elongating the previous note.
    """

    for input_path, output_path in input_output_dict.items():
        file_paths = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.mid') or file.endswith('.midi')]

        num_files = len(file_paths)
        note_arrays = {}
        logger.info(f"Processing {num_files} files in {input_path}")
        for i, path in enumerate(file_paths):
            try:
                note_array = midi_to_note_array(path, num_bars, start_bar_as_proportion=start_bar_proportion, remove_rests=remove_rests)
                note_arrays[path] = note_array
                logger.info(f"Processed {i+1}/{num_files} files")
            except Exception as e:
                logger.error(f"Error processing file {path}: {str(e)}")
                continue

        torch.save(note_arrays, output_path)

if __name__ == "__main__":
    monophonic_midis_to_note_arrays(
        {MAESTRO_PATH: 'data/exp1/maestro_one_bar_segments_nr.pt', MTC_PATH: 'data/exp1/mtc_one_bar_segments_nr.pt'},
        remove_rests=True
        )

