from typing import Tuple, Dict, Optional, List

from src.defaults.objects import NoteSequence

from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline, Splitter, Quantizer
from magenta.pipelines.melody_pipelines import MelodyExtractor

from note_seq.midi_io import midi_file_to_note_sequence
from note_seq.sequences_lib import extract_subsequence

def _calculate_two_bar_duration(note_sequence) -> float:
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
    hop_size_seconds = _calculate_two_bar_duration(ns)
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
    
def extract_2_bar_melodies(note_sequence) -> List[NoteSequence]:
    """
    Extract all possible melodies from a 2bar NoteSequence object, using the magenta MelodyExtractor.
    """
    ME = MelodyExtractor(min_bars=0, min_unique_pitches=1,gap_bars=2)
    melodies = ME.transform(note_sequence)
    mels = []
    for m in melodies:
        mels.append(m.to_sequence())
    return mels