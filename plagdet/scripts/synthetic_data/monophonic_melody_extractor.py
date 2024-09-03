# class for extracting monophonic melodies from a MIDI file
import logging
import mido
import pretty_midi
from mido import MidiFile, MidiTrack, MetaMessage
from magenta.models.music_vae import data
import magenta.music as mm

from plagdet.src.utils.log import configure_logging
from plagdet.scripts.synthetic_data.utils import calculate_total_bars, get_three_min_bars_from_file, get_tempo_and_time_signature, set_midi_tempo

configure_logging()
logger = logging.getLogger(__name__)

class MonophonicMelodyExtractor:

    def __init__(self):
        pass

    def sanitise_midi_tempo(self, file: str, dest_file: str):
        """
        Removes duplicate tempo messages from a MIDI file.
        """
        mid = MidiFile(file)
        new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
        tempo_set = False

        for track in mid.tracks:
            new_track = MidiTrack()
            for msg in track:
                if msg.type == 'set_tempo':
                    if not tempo_set:
                        new_track.append(msg)
                        tempo_set = True
                        logger.info(f"Kept first tempo: {mido.tempo2bpm(msg.tempo):.2f} BPM")
                    else:
                        logger.info(f"Removed additional tempo: {mido.tempo2bpm(msg.tempo):.2f} BPM")
                else:
                    new_track.append(msg)
            new_mid.tracks.append(new_track)

        new_mid.save(dest_file)
        logger.info(f"Sanitized MIDI file: {file}")

    def set_converter(self, bars: int):
        """
        Sets the converter for melody extraction based on the 
        """
        self.converter = data.OneHotMelodyConverter(
            valid_programs=data.MEL_PROGRAMS,
            skip_polyphony=False,
            max_bars=bars+2,  # Truncate long melodies before slicing.
            slice_bars=bars,
            steps_per_quarter=4)

    def extract_melodies(self, file: str) -> mm.NoteSequence:
        ns = mm.midi_file_to_note_sequence(file)
        melodies = self.converter.from_tensors(self.converter.to_tensors(ns)[1])
        return melodies

    def melody_to_file(self, melody, file):
        mm.note_sequence_to_midi_file(melody, file)

    def make_file_valid(self, file: str, dest_file: str):

        self.sanitise_midi_tempo(file, 'placeholder.midi')
        file = 'placeholder.midi'
        song_bars, _, _ = calculate_total_bars(file)
        three_min_bars = get_three_min_bars_from_file(file)

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

        tempo, _ = get_tempo_and_time_signature(file)
        self.set_converter(num_bars)
        if self.extract_melodies(file):
            melody = self.extract_melodies(file)[0]
        else:
            logger.info(f'No melody found in {file}')
            return
        
        self.melody_to_file(melody, dest_file)
        set_midi_tempo(dest_file, tempo)