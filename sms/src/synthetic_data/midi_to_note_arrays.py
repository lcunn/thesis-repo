import logging
from typing import Optional, List
import partitura as pt
import numpy as np

from sms.src.log import configure_logging

logger = logging.getLogger(__name__)
configure_logging()

def midi_to_note_array(
        midi_path: str, 
        num_bars: int, 
        start_bar: Optional[int] = None,
        start_bar_as_proportion: Optional[float] = None,
        rest_pitch: float = -1,
        remove_rests: bool = False
    ) -> np.ndarray:
    """
    Extracts bars from a MIDI file.
    We assume a bar is 4 beats, regardless of the time signature, for consistency across files.
    Exactly one of start_bar and start_bar_as_proportion must be provided.

    Args:
        midi_path (str): Path to the MIDI file.
        num_bars (int): Number of bars to extract.
        start_bar (int): Starting bar number.
        start_bar_as_proportion (float): Starting bar as a proportion of the total number of bars.
        rest_pitch (float): Pitch value for rests.
        remove_rests (bool): Whether to remove rests from the note array. 'Removes' rests by elongating the previous note.
    Returns:
        np.ndarray: Array with columns [duration_beat, pitch].
    """
    score = pt.load_score(midi_path)

    # ensure file not empty
    if not score.parts:
        raise ValueError("No parts found in the score.")
    part = score.parts[0]

    # we extract 4 beats each time, regardless of the time signature
    beats_per_bar = 4

    note_arr = part.note_array()
    last_note = note_arr[-1]
    last_note_end = last_note[0] + last_note[1]
    # if there is an incomplete bar at the end, we discount it
    bars_in_song = np.floor(last_note_end/4)

    # calculate start beats
    if start_bar is not None:
        start_beats = start_bar * beats_per_bar
    elif start_bar_as_proportion is not None:
        start_beats = start_bar_as_proportion * bars_in_song * beats_per_bar
    else:
        raise ValueError("Exactly one of start_bar and start_bar_as_proportion must be provided.")

    # calculate start and end beats
    end_beats = start_beats + num_bars * beats_per_bar

    # filter notes that start within the range or overlap the start
    extracted_notes = note_arr[
        (note_arr['onset_beat'] < end_beats) &
        (note_arr['onset_beat'] + note_arr['duration_beat'] > start_beats)
    ]

    if len(extracted_notes) <= 3:
        logger.info(f"Extracted notes: {extracted_notes}. Returning None.")
        return None

    # ensure notes sorted by onset_beat
    extracted_notes = extracted_notes[np.argsort(extracted_notes['onset_beat'])]

    # initialize list for duration and pitch
    duration_pitch = []
    previous_end = start_beats

    for note in extracted_notes:
        note_onset = note['onset_beat']
        note_duration = note['duration_beat']
        note_pitch = note['pitch']

        # adjust onset if the note starts before the start_beats
        adjusted_onset = max(note_onset, start_beats)
        
        # calculate actual duration within the range
        actual_duration = min(note_onset + note_duration, end_beats) - adjusted_onset

        # if there was a rest
        if adjusted_onset > previous_end:
            rest_duration = adjusted_onset - previous_end
            if remove_rests:
                # elongates previous note to account for the rest
                actual_duration += rest_duration
            else:
                duration_pitch.append([rest_duration, rest_pitch])

        # append note with adjusted duration
        duration_pitch.append([actual_duration, note_pitch])
        previous_end = adjusted_onset + actual_duration

    # handle end rest
    if previous_end < end_beats:
        rest_duration = end_beats - previous_end
        if remove_rests:
            # elongates last note to account for the rest
            duration_pitch[-1][0] += rest_duration
        else:
            duration_pitch.append([rest_duration, rest_pitch])

    # Convert to numpy array
    duration_pitch_array = np.array(duration_pitch, dtype=float)

    # Round the duration values to 3 decimal points
    duration_pitch_array[:, 0] = np.round(duration_pitch_array[:, 0], decimals=3)

    # validate that the sum of the durations is equal to beats_per_bar
    total_duration = np.sum(duration_pitch_array[:, 0])
    expected_duration = num_bars * beats_per_bar
    if not np.isclose(total_duration, expected_duration, atol=1e-8):
        logger.warning(f"Total duration {total_duration} does not match expected duration {expected_duration}")
        # adjust the last note/rest to make the total duration exactly 4
        duration_diff = expected_duration - total_duration
        duration_pitch_array[-1, 0] += duration_diff
        logger.info(f"Adjusted last note/rest duration by {duration_diff} to match expected duration")

    return duration_pitch_array

def midi_to_all_bars_efficient(
    midi_path: str,
    rest_pitch: float = -1,
    remove_rests: bool = True
) -> List[np.ndarray]:
    """
    Efficiently extracts all bars from a MIDI file and returns them as a list of note arrays.

    Args:
        midi_path (str): Path to the MIDI file.
        rest_pitch (float): Pitch value for rests.
        remove_rests (bool): Whether to remove rests from the note arrays.

    Returns:
        List[np.ndarray]: List of arrays, each representing a bar with columns [duration_beat, pitch].
    """
    score = pt.load_score(midi_path)

    if not score.parts:
        raise ValueError("No parts found in the score.")
    
    part = score.parts[0]
    note_arr = part.note_array()
    
    last_note = note_arr[-1]
    last_note_end = last_note['onset_beat'] + last_note['duration_beat']
    total_bars = int(np.ceil(last_note_end / 4))  # Assuming 4 beats per bar

    all_bars = []
    beats_per_bar = 4

    for bar in range(total_bars):
        start_beats = bar * beats_per_bar
        end_beats = start_beats + beats_per_bar

        # Filter notes for this bar
        bar_notes = note_arr[
            (note_arr['onset_beat'] < end_beats) &
            (note_arr['onset_beat'] + note_arr['duration_beat'] > start_beats)
        ]

        if len(bar_notes) < 2:
            logger.info(f"Bar {bar} has 1 note. Skipping.")
            continue

        # Sort notes by onset_beat
        bar_notes = bar_notes[np.argsort(bar_notes['onset_beat'])]

        # Process notes for this bar
        duration_pitch = []
        previous_end = start_beats

        for note in bar_notes:
            note_onset = note['onset_beat']
            note_duration = note['duration_beat']
            note_pitch = note['pitch']

            adjusted_onset = max(note_onset, start_beats)
            actual_duration = min(note_onset + note_duration, end_beats) - adjusted_onset

            if adjusted_onset > previous_end:
                rest_duration = adjusted_onset - previous_end
                if remove_rests:
                    actual_duration += rest_duration
                else:
                    duration_pitch.append([rest_duration, rest_pitch])

            duration_pitch.append([actual_duration, note_pitch])
            previous_end = adjusted_onset + actual_duration

        # Handle end rest
        if previous_end < end_beats:
            rest_duration = end_beats - previous_end
            if remove_rests:
                duration_pitch[-1][0] += rest_duration
            else:
                duration_pitch.append([rest_duration, rest_pitch])

        # Convert to numpy array and round durations
        bar_array = np.array(duration_pitch, dtype=float)
        bar_array[:, 0] = np.round(bar_array[:, 0], decimals=3)

        # Validate total duration
        total_duration = np.sum(bar_array[:, 0])
        if not np.isclose(total_duration, beats_per_bar, atol=1e-8):
            logger.warning(f"Bar {bar}: Total duration {total_duration} does not match expected duration {beats_per_bar}")
            duration_diff = beats_per_bar - total_duration
            bar_array[-1, 0] += duration_diff
            logger.info(f"Adjusted last note/rest duration by {duration_diff} to match expected duration")

        all_bars.append(bar_array)

    return all_bars