import collections

from magenta.common import merge_hparams
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
from magenta.models.music_vae import data_hierarchical
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE
import note_seq

HParams = contrib_training.HParams

class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name'])):

  def values(self):
    return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)

MEL_2BAR_CUSTOM_CONFIG = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
            free_bits=0,
            max_beta=0.5,
            beta_rate=0.99999,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=4,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)