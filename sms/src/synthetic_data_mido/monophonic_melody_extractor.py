# class for extracting monophonic melodies from a MIDI file
import logging
import os
import pandas as pd
import mido
import pretty_midi
import magenta.music as mm
from mido import MidiFile, MidiTrack, MetaMessage
from magenta.models.music_vae import data

from sms.defaults import *
from sms.src.log import configure_logging
from sms.src.synthetic_data_mido.utils import calculate_total_bars, get_midi_info, calculate_bars_for_x_minutes, set_midi_tempo

configure_logging()
logger = logging.getLogger(__name__)

class MonophonicMelodyExtractor:

    def __init__(self):
        pass

    # def sanitise_midi_tempo(self, file: str, dest_file: str):
    #     """
    #     Removes duplicate tempo messages from a MIDI file.
    #     """
    #     mid = MidiFile(file)
    #     new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    #     tempo_set = False

    #     for track in mid.tracks:
    #         new_track = MidiTrack()
    #         for msg in track:
    #             if msg.type == 'set_tempo':
    #                 if not tempo_set:
    #                     new_track.append(msg)
    #                     tempo_set = True
    #                     logger.info(f"Kept first tempo: {mido.tempo2bpm(msg.tempo):.2f} BPM")
    #                 else:
    #                     logger.info(f"Removed additional tempo: {mido.tempo2bpm(msg.tempo):.2f} BPM")
    #             else:
    #                 new_track.append(msg)
    #         new_mid.tracks.append(new_track)

    #     new_mid.save(dest_file)
    #     logger.info(f"Sanitized MIDI file: {file}")

    def set_converter(self, bars: int):
        """
        Sets the converter for melody extraction based on the 
        """
        self.converter = data.OneHotMelodyConverter(
            valid_programs=data.MEL_PROGRAMS,
            skip_polyphony=False,
            max_bars=bars+2,  # Truncate long melodies before slicing.
            slice_bars=bars,
            steps_per_quarter=4, 
            gap_bars=2)

    def extract_melodies(self, file: str) -> mm.NoteSequence:
        ns = mm.midi_file_to_note_sequence(file)
        melodies = self.converter.from_tensors(self.converter.to_tensors(ns)[1])
        return melodies

    def melody_to_file(self, melody, file):
        mm.note_sequence_to_midi_file(melody, file)

    def make_file_valid(self, file: str, dest_file: str) -> bool:

        # self.sanitise_midi_tempo(file, 'placeholder.midi')
        # file = 'placeholder.midi'
        song_bars = calculate_total_bars(file)
        three_min_bars = calculate_bars_for_x_minutes(file, 3)

        above_min = song_bars/three_min_bars > 0.75
        more_than_three = song_bars > three_min_bars

        if more_than_three:
            num_bars = three_min_bars
            logger.info(f'{file} is longer than 3 minutes, extracting {num_bars} bars.')
        elif above_min:
            num_bars = song_bars
            logger.info(f'{file} is longer than 0.75 minutes, extracting {num_bars} bars.')
        else:
            logger.warning(f'{file} is too short to extract a melody.')
            return

        tempo, _, _ = get_midi_info(file)
        self.set_converter(int(num_bars))
        if self.extract_melodies(file):
            melody = self.extract_melodies(file)[0]
        else:
            logger.info(f'No melody found in {file}')
            return False
        
        self.melody_to_file(melody, dest_file)
        set_midi_tempo(dest_file, tempo)
        return True

def produce_monophonic_dataset(evaluable_directory_path: str) -> None:

    extractor = MonophonicMelodyExtractor()
    directory_path = eval(evaluable_directory_path)
    
    evaluable_destination_path = 'MONOPHONIC_MIDIS_PATH'
    destination_path = eval(evaluable_destination_path)

    # Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)

    if not os.path.exists(PROCESSED_MELODY_TRACKER):
        log = pd.DataFrame(columns=['raw_root', 'raw_relative_path', 'success', 'mono_root', 'mono_relative_path'])
    else:
        log = pd.read_csv(PROCESSED_MELODY_TRACKER)

    for root, _, files in os.walk(directory_path):
        for file in files:
            path = os.path.join(root, file)
            relative_path = os.path.relpath(path, directory_path)
            # Check if the file is a MIDI file
            if not file.lower().endswith(('.mid', '.midi')):
                logger.info(f'Skipping {file} as it is not a MIDI file')
                continue
            
            if relative_path in log['raw_relative_path'].values:
                continue

            logger.info(f'Processing {path}')
            dest = os.path.join(destination_path, f'{os.path.splitext(file)[0]}_mono.mid')
            success = extractor.make_file_valid(path, dest)
            
            new_row = pd.DataFrame({
                'raw_root': [evaluable_directory_path],
                'raw_relative_path': [relative_path],
                'success': [success],
                'mono_root': [evaluable_destination_path],
                'mono_relative_path': [os.path.relpath(dest, destination_path) if success else None]
            })
            log = pd.concat([log, new_row], ignore_index=True)

            if success:
                logger.info(f'{file} successfully made monophonic')

    log.to_csv(PROCESSED_MELODY_TRACKER, index=False)
    
if __name__ == '__main__':
    produce_monophonic_dataset('MAESTRO_PATH')