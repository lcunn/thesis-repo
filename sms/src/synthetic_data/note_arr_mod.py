import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from sms.src.log import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

@dataclass
class NoteArrayModifierConfig:

    use_transposition: bool = False
    transposition_semitone: Optional[int] = None

    use_shift_selected_notes_pitch: bool = False
    # tuple of (note_idx, semitone_shift):
    selected_notes_pitch_shifts: Optional[List[Tuple[int, int]]] = field(default_factory=list)

    use_change_note_durations: bool = False
    # tuple of (note_idx, scale_factor):
    note_scale_factors: Optional[List[Tuple[int, float]]] = field(default_factory=list)

    use_delete_notes: bool = False
    # list of note_idx:
    notes_to_delete: Optional[List[int]] = field(default_factory=list)

@dataclass
class NoteArrayModifierSettings:
    transposition_semitone_range: Tuple[int, int] = (-16, 16)
    notes_to_pitch_shift: int = 1
    note_pitch_shift_range: Tuple[int, int] = (-12, 12)
    notes_to_scale: int = 1
    note_duration_scale_options: Tuple[float, ...] = (0.5, 1.5, 2, 3)
    notes_to_delete: int = 1

class NoteArrayModifier:
    def __init__(self, settings: NoteArrayModifierSettings = NoteArrayModifierSettings()):
        self.note_array: Optional[np.ndarray] = None
        self.config: Optional[NoteArrayModifierConfig] = None
        self.settings: NoteArrayModifierSettings = settings

    def set_note_array(self, note_array: np.ndarray):
        if note_array.ndim != 2 or note_array.shape[1] != 2:
            raise ValueError("note_array must be a 2D array with 2 columns [duration_beat, pitch].")
        self.note_array = note_array.copy()

    def generate_and_set_config(
        self,
        use_transposition: bool = False,
        use_shift_selected_notes_pitch: bool = False,
        use_change_note_durations: bool = False,
        use_delete_notes: bool = False
    ) -> None:
        if self.note_array is None:
            raise ValueError("Note array is not set.")

        num_notes = self.note_array.shape[0]
        config = NoteArrayModifierConfig(
            use_transposition=use_transposition,
            use_shift_selected_notes_pitch=use_shift_selected_notes_pitch,
            use_change_note_durations=use_change_note_durations,
            use_delete_notes=use_delete_notes
        )

        if use_transposition:
            # randomly choose a semitone value from the range
            config.transposition_semitone = int(
                np.random.randint(
                    self.settings.transposition_semitone_range[0],
                    self.settings.transposition_semitone_range[1]
                )
            )
        
        if use_shift_selected_notes_pitch:
            # get indices of non-rest notes (pitch != 0)
            non_rest_indices = np.where(self.note_array[:, 1] != 0)[0]
            num_non_rest = len(non_rest_indices)
            num_shifts = min(self.settings.notes_to_pitch_shift, num_non_rest)
            if num_shifts > 0:
                # randomly choose num_shifts indices from the non-rest notes
                shift_indices = np.random.choice(non_rest_indices, num_shifts, replace=False)
                # randomly choose a semitone value from the range
                shifts = np.random.randint(
                    self.settings.note_pitch_shift_range[0],
                    self.settings.note_pitch_shift_range[1],
                    size=num_shifts
                )
                config.selected_notes_pitch_shifts = list(zip(shift_indices, shifts))
        
        if use_change_note_durations:
            num_scales = min(self.settings.notes_to_scale, num_notes)
            if num_scales > 0:
                # randomly choose num_scales indices from the note array
                scale_indices = np.random.choice(num_notes, num_scales, replace=False)
                # randomly choose a scale factor from the options
                scales = np.random.choice(
                    self.settings.note_duration_scale_options,
                    size=num_scales
                )
                config.note_scale_factors = list(zip(scale_indices, scales))
        
        if use_delete_notes:
            num_deletes = min(self.settings.notes_to_delete, num_notes)
            if num_deletes > 0:
                # randomly choose num_deletes indices from the note array
                delete_indices = np.random.choice(num_notes, num_deletes, replace=False)
                config.notes_to_delete = list(delete_indices)
        
        self.config = config

    def modify_note_array(self) -> None:
        if self.note_array is None or self.config is None:
            raise ValueError("Note array or config is not set.")

        original_total_duration = np.sum(self.note_array[:, 0])
        modified_array = self.note_array.copy()

        if self.config.use_transposition and self.config.transposition_semitone is not None:
            # apply transposition to all non-rest notes
            non_rest_mask = modified_array[:, 1] != 0
            modified_array[non_rest_mask, 1] += self.config.transposition_semitone
            logger.info(f'Transposing non-rest notes by {self.config.transposition_semitone} semitones.')

        if self.config.use_shift_selected_notes_pitch and self.config.selected_notes_pitch_shifts:
            for idx, shift in self.config.selected_notes_pitch_shifts:
                modified_array[idx, 1] += shift
                logger.info(f'Shifting note at index {idx} by {shift} semitones.')

        if self.config.use_change_note_durations and self.config.note_scale_factors:
            for idx, scale in self.config.note_scale_factors:
                modified_array[idx, 0] *= scale
                logger.info(f'Scaling duration of note at index {idx} by a factor of {scale}.')

        if self.config.use_delete_notes and self.config.notes_to_delete:
            modified_array = np.delete(modified_array, self.config.notes_to_delete, axis=0)
            logger.info(f'Deleting notes at indices {self.config.notes_to_delete}.')

        # ensure total duration remains the same
        modified_total_duration = np.sum(modified_array[:, 0])
        difference = original_total_duration - modified_total_duration

        # if sequence now shorter
        if difference > 0:
            # add a rest at the end
            modified_array = np.vstack([modified_array, [difference, 0]])
            logger.info(f'Added a rest of duration {difference} to maintain total duration.')
        elif difference < 0:
            # cut off/truncate notes until total duration is reached
            remaining_diff = -difference
            i = len(modified_array) - 1
            while remaining_diff > 0 and i >= 0:
                current_duration = modified_array[i, 0]
                if current_duration > remaining_diff:
                    modified_array[i, 0] -= remaining_diff
                    logger.info(f'Truncated note {i} by {remaining_diff} to maintain total duration.')
                    remaining_diff = 0
                else:
                    remaining_diff -= current_duration
                    logger.info(f'Removed note {i} with duration {current_duration} to adjust total duration.')
                    modified_array = np.delete(modified_array, i, axis=0)
                    i -= 1
            if remaining_diff > 0:
                # add a rest for any remaining difference
                modified_array = np.vstack([modified_array, [remaining_diff, 0]])
                logger.info(f'Added a rest of duration {remaining_diff} to maintain total duration.')

        self.note_array = modified_array

    def get_modified_note_array(self) -> np.ndarray:
        if self.note_array is None:
            raise ValueError("Note array is not set.")
        return self.note_array