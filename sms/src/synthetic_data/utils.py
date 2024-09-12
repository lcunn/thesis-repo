# class for extracting monophonic melodies from a MIDI file
from mido import MidiFile, MidiTrack, MetaMessage
import mido
import pretty_midi
from magenta.models.music_vae import data
import magenta.music as mm
from typing import Optional, Tuple
import logging

from sms.src.log import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def get_midi_info(midi_file_path):
    """
    Returns the tempo, time signature and ticks per beat of a MIDI file.
    """
    midi = MidiFile(midi_file_path)
    tempo = None
    time_signature = (4, 4)  # Default
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'time_signature':
                time_signature = (msg.numerator, msg.denominator)
            if tempo and time_signature != (4, 4):
                break
        if tempo and time_signature != (4, 4):
            break
    return tempo, time_signature, midi.ticks_per_beat

def calculate_bar_duration(tempo, time_signature, ticks_per_beat):
    microseconds_per_beat = tempo
    beats_per_minute = 60000000 / microseconds_per_beat
    beats_per_bar = time_signature[0]
    seconds_per_beat = 60 / beats_per_minute
    return seconds_per_beat * beats_per_bar

def calculate_bars_for_x_minutes(midi_file_path: str, x: float) -> float:
    tempo, time_signature, _ = get_midi_info(midi_file_path)
    bpm = mido.tempo2bpm(tempo)
    beats_per_bar = time_signature[0]
    total_beats = bpm * x 
    total_bars = total_beats / beats_per_bar
    return total_bars

def calculate_total_bars(midi_file_path) -> float:
    tempo, time_signature, ticks_per_beat = get_midi_info(midi_file_path)
    midi = MidiFile(midi_file_path)
    
    total_ticks = 0
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta:
                total_ticks += msg.time
    
    total_beats = total_ticks / ticks_per_beat
    beats_per_bar = time_signature[0]  # Numerator gives the number of beats per bar
    total_bars = total_beats / beats_per_bar
    return total_bars

def set_midi_tempo(file: str, tempo: float) -> None:

    mid = MidiFile(file)
    new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    
    tempo_set = False
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type == 'set_tempo' and not tempo_set:
                new_track.append(MetaMessage('set_tempo', tempo=tempo, time=msg.time))
                tempo_set = True
                logger.info(f"Adjusted tempo to original: {tempo} BPM")
            else:
                new_track.append(msg)
        new_mid.tracks.append(new_track)
    
    if not tempo_set:
        # If no tempo message was found, add it to the first track
        new_tempo = mido.bpm2tempo(tempo)
        new_mid.tracks[0].insert(0, MetaMessage('set_tempo', tempo=new_tempo, time=0))
        logger.info(f"Added original tempo: {tempo} BPM")

    new_mid.save(file)