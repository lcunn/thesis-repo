import os
import torch
import logging
from typing import Dict, List
import numpy as np
import sys
import mido

from sms.src.synthetic_data.midi_to_note_arrays import midi_to_all_bars_efficient
from sms.src.synthetic_data.note_arr_mod import NoteArrayModifier, NoteArrayModifierSettings
from sms.src.log import configure_logging
from sms.defaults import MAESTRO_PATH, MTC_PATH, MAESTRO_SEGMENTS_PATH, MTC_SEGMENTS_PATH

logger = logging.getLogger(__name__)
configure_logging(console_level=logging.INFO)

def scale_midi_tempo(mid: mido.MidiFile, scale: float) -> mido.MidiFile:
    """
    Scale the tempo of a MIDI file by adjusting note durations.
    
    Args:
        mid (mido.MidiFile): The input MIDI file.
        scale (float): The scaling factor for tempo.
    
    Returns:
        mido.MidiFile: A new MIDI file with scaled note durations.
    """
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            new_msg = msg.copy()
            if hasattr(msg, 'time'):
                new_msg.time = int(msg.time / scale)
            new_track.append(new_msg)
        new_mid.tracks.append(new_track)
    
    return new_mid

def extract_all_chunks_from_dir(
        input_dir: str, 
        output_file: str,
        tempo_scales: List[float] = [0.5, 1.0, 2.0]
        ) -> None:
    """
    Takes directories of MIDI files, applies tempo scaling, and converts all bars from every song to note arrays.

    Args:
        input_dir (str): Input directory path.
        output_file (str): Path to save the output file.
        tempo_scales (List[float]): List of tempo scaling factors to apply.
    """
    all_chunks = {}

    file_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.mid') or file.endswith('.midi')]
    num_files = len(file_paths)
    logger.info(f"Processing {num_files} files in {input_dir}")

    for i, path in enumerate(file_paths):
        try:
            file_chunks = {}
            original_mid = mido.MidiFile(path)
            
            for scale in tempo_scales:
                # Scale the tempo
                scaled_mid = scale_midi_tempo(original_mid, scale)
                scaled_mid.save("temp.midi")
                
                # Convert to note arrays
                scaled_note_arrays = midi_to_all_bars_efficient("temp.midi")
                file_chunks[scale] = scaled_note_arrays
            
            all_chunks[os.path.basename(path)] = file_chunks
            logger.info(f"Processed {i+1}/{num_files} files")
        except Exception as e:
            logger.error(f"Error processing file {path}: {str(e)}")
            continue

    os.remove("temp.midi")

    torch.save(all_chunks, output_file)
    logger.info(f"Saved chunks from {len(all_chunks)} songs to {output_file}")

if __name__ == "__main__":
    os.chdir(r"C:\Users\cunn2\OneDrive\DSML\Project\thesis-repo")
    input_dir = r"data\exp3\mpd_mid"
    output_file = r"data\exp3\all_plag_chunks.pt"
    extract_all_chunks_from_dir(input_dir, output_file)