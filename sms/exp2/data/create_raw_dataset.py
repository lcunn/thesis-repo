import os
import torch
import logging
from typing import Dict, List
import numpy as np
import sys

from sms.src.synthetic_data.midi_to_note_arrays import midi_to_all_bars_efficient
from sms.src.synthetic_data.note_arr_mod import NoteArrayModifier, NoteArrayModifierSettings
from sms.src.log import configure_logging
from sms.defaults import MAESTRO_PATH, MTC_PATH, MAESTRO_SEGMENTS_PATH, MTC_SEGMENTS_PATH

logger = logging.getLogger(__name__)
configure_logging(console_level=logging.INFO)

def extract_all_chunks_from_dirs(
        input_dirs: List[str], 
        output_file: str
        ) -> None:
    """
    Takes directories of MIDI files and converts all bars from every song to note arrays.

    Args:
        input_dirs (List[str]): List of input directory paths.
        output_file (str): Path to save the output file.
    """
    all_chunks = {}

    for input_dir in input_dirs:
        file_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.mid') or file.endswith('.midi')]
        num_files = len(file_paths)
        logger.info(f"Processing {num_files} files in {input_dir}")

        for i, path in enumerate(file_paths):
            try:
                note_arrays = midi_to_all_bars_efficient(path)
                all_chunks[os.path.basename(path)] = note_arrays
                logger.info(f"Processed {i+1}/{num_files} files")
            except Exception as e:
                logger.error(f"Error processing file {path}: {str(e)}")
                continue

    torch.save(all_chunks, output_file)
    logger.info(f"Saved chunks from {len(all_chunks)} songs to {output_file}")

    # Estimate RAM usage
    ram_usage = sys.getsizeof(all_chunks) / (1024 * 1024)  # Convert to MB
    logger.info(f"Estimated RAM usage: {ram_usage:.2f} MB")

def augment_all_note_arrays(
        input_file: str, 
        output_file: str, 
        num_augmentations: int, 
        total_songs: int
        ) -> None:
    """
    Augments note arrays from the input file and saves the result to the output file.

    Args:
        input_file (str): Path to the input file containing note arrays.
        output_file (str): Path to save the augmented note arrays.
        num_augmentations (int): Number of times to augment each chunk.
        total_songs (int): Total number of augmented songs to generate.
    """
    settings = NoteArrayModifierSettings(
        transposition_semitone_range=(-4, 4),
        notes_to_pitch_shift=1,
        note_pitch_shift_range=(-4, 4),
        notes_to_scale=1,
        note_duration_scale_options=(0.5, 1.5, 2),
        notes_to_delete=1,
        notes_to_insert=1,
        insert_note_duration_options=(0.25, 0.5),
        insert_note_relative_pitch_range=(-4, 4)
    )

    aug_dict = {
        "use_transposition": True,
        "use_shift_selected_notes_pitch": True,
        "use_change_note_durations": True,
        "use_delete_notes": True,
        "use_insert_notes": True
    }

    modifier = NoteArrayModifier(settings=settings)

    input_chunks = torch.load(input_file)
    augmented_chunks = {}
    
    song_names = list(input_chunks.keys())
    augmented_count = 0

    # while augmented_count < total_songs:
    #     song_name = np.random.choice(song_names)
    #     chunk = np.random.choice(input_chunks[song_name])
    #     for _ in range(num_augmentations):
    #         if augmented_count >= total_songs:
    #             break
    #         augmented_chunk = modifier.modify_note_array(chunk, **aug_dict)
    #         aug_song_name = f"{song_name}_aug_{augmented_count}"
    #         augmented_chunks[aug_song_name] = augmented_chunk
    #         augmented_count += 1

    for song_name, chunks in input_chunks.items():
        if not isinstance(chunks, (list, np.ndarray)):
            chunks = [chunks]  # Convert single chunk to list
        
        for i, chunk in enumerate(chunks):
            for j in range(num_augmentations):
                if augmented_count >= total_songs:
                    break
                
                augmented_chunk = modifier(chunk, aug_dict)
                aug_song_name = f"{song_name}_chunk{i}_aug{j}"
                augmented_chunks[aug_song_name] = augmented_chunk
                augmented_count += 1
            
            if augmented_count >= total_songs:
                break
        
        if augmented_count >= total_songs:
            break

    torch.save(augmented_chunks, output_file)
    logger.info(f"Saved {len(augmented_chunks)} augmented chunks to {output_file}")

    # Estimate RAM usage
    ram_usage = sys.getsizeof(augmented_chunks) / (1024 * 1024)  # Convert to MB
    logger.info(f"Estimated RAM usage: {ram_usage:.2f} MB")

if __name__ == "__main__":
    # Extract all chunks from MIDI files
    if not os.path.exists('data/exp2/all_chunks.pt'):
        extract_all_chunks_from_dirs([MAESTRO_PATH, MTC_PATH], 'data/exp2/all_chunks.pt')

    # Augment the extracted chunks
    if not os.path.exists('data/exp2/augmented_chunks.pt'):
        augment_all_note_arrays('data/exp2/all_chunks.pt', 'data/exp2/augmented_chunks.pt', num_augmentations=4, total_songs=1_000_000)

