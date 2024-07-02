from typing import str

from src.midi.midi_processor import MidiProcessor

from note_seq.sequences_lib import extract_subsequence
from note_seq.midi_io import midi_file_to_note_sequence, note_sequence_to_midi_file
from magenta.music import sequences_lib
from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline, Splitter, Quantizer

"""
Using the encoder from the notebook...



"""


class MusicVAEncoder(MidiProcessor):
    def __init__(self, file_path: str, args):
        super().__init__(file_path)
        self._note_sequence = midi_file_to_note_sequence(file_path)
        self.args = args  #what are these args?


    def _calculate_two_bar_duration(self):
        if self._note_sequence.tempos:
            bpm = self._note_sequence.tempos[0].qpm  # Using the first tempo for calculation
        else:
            bpm = 120  # Default tempo
        
        if self._note_sequence.time_signatures:
            beats_per_bar = self._note_sequence.time_signatures[0].numerator  # Assuming simple time signatures
        else:
            beats_per_bar = 4  # Common time
        
        seconds_per_beat = 60 / bpm
        return 2 * beats_per_bar * seconds_per_beat  # Duration of two bars
    
    def split_into_2_bar_segments(midi, quantize=True, steps_per_quarter=8):
        """
        Takes a midi file and splits it into a list of 2-bar NoteSequence objects with a window shift of 1 bar.

        Args:
            midi: path to the midi file.
            quantize: binary, whether to quantize or not.
            steps_per_quarter: argument for the Quantizer instance.

        Returns:
            List of NoteSequence objects.
        """
        ns = midi_file_to_note_sequence(midi)

        quantizer = Quantizer(steps_per_quarter=8)

        # Calculate the duration of two bars
        hop_size_seconds = calculate_two_bar_duration(ns)

        # Split the NoteSequence into segments of two bars each
        splitter = Splitter(hop_size_seconds=hop_size_seconds)
        two_bar_segments = splitter.transform(ns)
        
        # Get the 2 bars shifted 1 bar forwards
        shifted_ns = extract_subsequence(ns, hop_size_seconds/2, ns.total_time)
        shifted_two_bar_segments = splitter.transform(shifted_ns)
        
        two_bar_segments = two_bar_segments + shifted_two_bar_segments

        if quantize:
            # loop over and quantize each segment
            two_bar_segments_quant = two_bar_segments

            for i, segment in enumerate(two_bar_segments):
                two_bar_segments_quant[i] = quantizer.transform(segment)[0]
            return two_bar_segments_quant
        
        else:
            return two_bar_segments
