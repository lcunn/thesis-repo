# class for extracting monophonic melodies from a MIDI file
from mido import MidiFile, MidiTrack, MetaMessage
import mido
import pretty_midi
from magenta.models.music_vae import data
import magenta.music as mm
from typing import Optional
import logging

from plagdet.src.utils.log import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def get_tempo_and_time_signature(midi_file_path: str):
    midi = MidiFile(midi_file_path)
    
    tempo = None
    time_signature = (4, 4)  # Default time signature is 4/4
    additional_tempos = []

    for track_idx, track in enumerate(midi.tracks):
        for msg_idx, msg in enumerate(track):
            if msg.type == 'set_tempo':
                if tempo is None:
                    tempo = msg.tempo  # Tempo is in microseconds per beat
                else:
                    additional_tempos.append((track_idx, msg_idx, msg.tempo))
            elif msg.type == 'time_signature':
                time_signature = (msg.numerator, msg.denominator)
    
    if tempo is not None:
        bpm = 60000000 / tempo  # Convert tempo to BPM
    else:
        bpm = 120  # Default to 120 BPM if no tempo is set

    if additional_tempos:
        logger.info(f"Additional tempo changes found in {midi_file_path}:")
        for track_idx, msg_idx, tempo in additional_tempos:
            logger.info(f"  Track {track_idx}, Message {msg_idx}: {60000000 / tempo:.2f} BPM")

    return bpm, time_signature

def calculate_bars_for_three_minutes(bpm, time_signature):
    beats_per_bar = time_signature[0]  # Numerator gives the number of beats per bar
    total_beats_in_three_minutes = bpm * 3  # 3 minutes = 180 seconds, hence multiply BPM by 3
    total_bars = total_beats_in_three_minutes / beats_per_bar
    return total_bars

def calculate_total_bars(midi_file_path):
    midi = MidiFile(midi_file_path)
    bpm, time_signature = get_tempo_and_time_signature(midi_file_path)
    
    ticks_per_beat = midi.ticks_per_beat
    total_ticks = 0

    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta:
                total_ticks += msg.time
    
    total_beats = total_ticks / ticks_per_beat
    beats_per_bar = time_signature[0]  # Numerator gives the number of beats per bar
    total_bars = total_beats / beats_per_bar

    return total_bars, total_beats, bpm

def get_three_min_bars_from_file(file: str) -> int:
    bpm, time_signature = get_tempo_and_time_signature(file)
    return int(calculate_bars_for_three_minutes(bpm, time_signature))

def set_midi_tempo(file: str, tempo: float) -> None:

    mid = MidiFile(file)
    new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    
    tempo_set = False
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type == 'set_tempo' and not tempo_set:
                new_tempo = mido.bpm2tempo(tempo)
                new_track.append(MetaMessage('set_tempo', tempo=new_tempo, time=msg.time))
                tempo_set = True
                logger.info(f"Adjusted tempo to original: {tempo} BPM")
            else:
                new_track.append(msg)
        new_mid.tracks.append(new_track)
    
    if not tempo_set:
        # If no tempo message was found, add it to the first track
        new_tempo = mido.bpm2tempo(tempo)
        new_mid.tracks[0].insert(0, MetaMessage('set_tempo', tempo=new_tempo, time=0))
        logger.info(f"Added original tempo: {self.tempo} BPM")

    new_mid.save(file)