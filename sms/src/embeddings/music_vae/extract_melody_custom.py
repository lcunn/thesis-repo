from typing import Tuple, Dict, Optional, List

from sms.defaults import *

from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline, Splitter, Quantizer
from magenta.pipelines.melody_pipelines import MelodyExtractor

from note_seq.midi_io import note_sequence_to_midi_file
from note_seq.midi_io import midi_file_to_note_sequence
from note_seq.sequences_lib import extract_subsequence

def calculate_two_bar_duration(note_sequence) -> float:
    """
    Calculates the length of 2 bars of a NoteSequence file in seconds.
    """
    # Use first tempo, else default
    bpm = note_sequence.tempos[0].qpm if note_sequence.tempos else 120
    # Assuming simple time signatures, else common time 
    beats_per_bar = note_sequence.time_signatures[0].numerator if note_sequence.time_signatures else 4    
    seconds_per_beat = 60 / bpm
    two_bar_duration = 2 * beats_per_bar * seconds_per_beat
    return two_bar_duration

def split_into_2_bar_segments(
        note_sequence, 
        quantize: bool = True,    
        quantize_steps_per_quarter: int = 8
        ) -> Tuple[NoteSequence]:
    """
    Takes a midi file and splits it into a list of 2-bar NoteSequence objects with a window shift of 1 bar.
    Args:
        note_sequence.
        quantize: whether to quantize or not.
        steps_per_quarter: argument for the Quantizer instance.
    Returns:
        Tuple of NoteSequence objects, indexed by bar position.
    """
    ns = note_sequence
    # Calculate the duration of two bars
    hop_size_seconds = calculate_two_bar_duration(ns)
    # Split the NoteSequence into segments of two bars each
    splitter = Splitter(hop_size_seconds=hop_size_seconds)
    two_bar_segments = splitter.transform(ns)
    # Get the 2 bars shifted 1 bar forwards
    shifted_ns = extract_subsequence(ns, hop_size_seconds/2, ns.total_time)
    shifted_two_bar_segments = splitter.transform(shifted_ns)
    two_bar_segments_all = [None for i in range(len(two_bar_segments)+len(shifted_two_bar_segments))]
    two_bar_segments_all[::2], two_bar_segments_all[1::2] = two_bar_segments, shifted_two_bar_segments
    if quantize:
        # loop over and quantize each segment
        quantizer = Quantizer(quantize_steps_per_quarter)
        for i, segment in enumerate(two_bar_segments_all):
            two_bar_segments_all[i] = quantizer.transform(segment)[0]
        return tuple(two_bar_segments_all)
    else:
        return tuple(two_bar_segments_all)
    
# def extract_melodies(quantized_sequence,
#                      search_start_step=0,
#                      min_bars=7,
#                      max_steps_truncate=None,
#                      max_steps_discard=None,
#                      gap_bars=1.0,
#                      min_unique_pitches=5,
#                      ignore_polyphonic_notes=True,
#                      pad_end=False,
#                      filter_drums=True):
#   """Extracts a list of melodies from the given quantized NoteSequence.

#   This function will search through `quantized_sequence` for monophonic
#   melodies in every track at every time step.

#   Once a note-on event in a track is encountered, a melody begins.
#   Gaps of silence in each track will be splitting points that divide the
#   track into separate melodies. The minimum size of these gaps are given
#   in `gap_bars`. The size of a bar (measure) of music in time steps is
#   computed from the time signature stored in `quantized_sequence`.

#   The melody is then checked for validity. The melody is only used if it is
#   at least `min_bars` bars long, and has at least `min_unique_pitches` unique
#   notes (preventing melodies that only repeat a few notes, such as those found
#   in some accompaniment tracks, from being used).

#   After scanning each instrument track in the quantized sequence, a list of all
#   extracted Melody objects is returned.

#   Args:
#     quantized_sequence: A quantized NoteSequence.
#     search_start_step: Start searching for a melody at this time step. Assumed
#         to be the first step of a bar.
#     min_bars: Minimum length of melodies in number of bars. Shorter melodies are
#         discarded.
#     max_steps_truncate: Maximum number of steps in extracted melodies. If
#         defined, longer melodies are truncated to this threshold. If pad_end is
#         also True, melodies will be truncated to the end of the last bar below
#         this threshold.
#     max_steps_discard: Maximum number of steps in extracted melodies. If
#         defined, longer melodies are discarded.
#     gap_bars: A melody comes to an end when this number of bars (measures) of
#         silence is encountered.
#     min_unique_pitches: Minimum number of unique notes with octave equivalence.
#         Melodies with too few unique notes are discarded.
#     ignore_polyphonic_notes: If True, melodies will be extracted from
#         `quantized_sequence` tracks that contain polyphony (notes start at
#         the same time). If False, tracks with polyphony will be ignored.
#     pad_end: If True, the end of the melody will be padded with NO_EVENTs so
#         that it will end at a bar boundary.
#     filter_drums: If True, notes for which `is_drum` is True will be ignored.
    
def extract_2_bar_melodies(note_sequence) -> List[NoteSequence]:
    """
    Extract all possible melodies from a 2bar NoteSequence object, using the magenta MelodyExtractor.
    """
    ME = MelodyExtractor(min_bars=0, 
                         min_unique_pitches=3,
                         gap_bars=2)
    melodies = ME.transform(note_sequence)
    mels = []
    for m in melodies:
        mels.append(m.to_sequence())
    return mels

def extract_melodies_custom(midi_path: str) -> Dict[str, NoteSequence]:
    """
    Converts midi to 2 bar NoteSequence segments.
    Extracts all melodies from the list of 2 bar segments using the magenta MelodyExtractor and stores in a dictionary.
    Keys are a string representation of the starting bar of the 2 bar melody, e.g. '0'.
    """
    note_sequence = midi_file_to_note_sequence(midi_path)
    two_bar_segments_q = split_into_2_bar_segments(note_sequence)

    melodies = {}
    for start_bar, segment in enumerate(two_bar_segments_q):
        mels = extract_2_bar_melodies(segment)
        melodies[str(start_bar)] = mels
    return melodies
