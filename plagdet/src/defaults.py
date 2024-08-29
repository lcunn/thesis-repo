from typing import List

NoteSequence = object

# Model paths

MUSICVAE_2BAR_BIG_CONFIG = 'cat-mel_2bar_big'
MUSICVAE_2BAR_BIG_PATH = 'plagdet/models/music_vae/2_bar_melody/cat-mel_2bar_big.ckpt'

# Database CSV file paths

COPYRIGHT_CLAIMS_CSV = 'plagdet/data/case_data/intermediate_csvs/all_copyright_claims.csv'
COPYRIGHT_CLAIMS_CSV_F = 'plagdet/data/case_data/intermediate_csvs/all_copyright_claims_filtered.csv'
GPT_SONGS_CHECKPOINT_CSV = 'plagdet/data/case_data/intermediate_csvs/gpt_songs_checkpoint.csv'
ESTIMATED_SONGS_CSV = 'plagdet/data/case_data/intermediate_csvs/estimated_songs.csv'

SONGS_TO_VALIDATE_CSV = 'plagdet/data/case_data/intermediate_csvs/songs_to_validate.csv'
VALIDATED_SONGS_CSV = 'plagdet/data/case_data/intermediate_csvs/validated_songs.csv'

COPYRIGHT_SONGS_CSV = 'plagdet/data/case_data/final_csvs/final_copyright_songs.csv'
COPYRIGHT_PAIRS_CSV = 'plagdet/data/case_data/final_csvs/final_copyright_pairs.csv'

# MIDI search log path

MIDI_SEARCH_LOG_PATH = 'plagdet/data/midi_search_logs'

# MIDI data paths

DATABASE = 'plagdet/data/music.db'
MIDI_DATA_PATH = 'plagdet/data/midi_databases'
COPYRIGHT_MIDI_PATH = 'plagdet/data/copyright_midis'

LAKH_CLEAN_PATH = 'plagdet/data/midi_databases/lakh_clean'

BIMMUDA_PATH = 'plagdet/data/midi_databases/bimmuda'
BIMMUDA_METADATA_PATH = 'plagdet/data/midi_databases/bimmuda/bimmuda_per_song_metadata.csv'

METAMIDI_PATH = 'plagdet/data/midi_databases/metamidi/MMD_MIDI'
METAMIDI_METADATA_PATH = 'plagdet/data/midi_databases/metamidi/MMD_scraped_title_artist.jsonl'

MPD_PATH = 'plagdet/data/midi_databases/MPDSet'

