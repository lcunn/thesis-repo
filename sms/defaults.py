from typing import List

NoteSequence = object

# Model paths

MUSICVAE_2BAR_BIG_CONFIG = 'cat-mel_2bar_big'
MUSICVAE_2BAR_BIG_PATH = 'models/music_vae/2_bar_melody/cat-mel_2bar_big.ckpt'

# Database CSV file paths

COPYRIGHT_CLAIMS_CSV = 'data/plagiarism_dataset/case_data/intermediate_csvs/all_copyright_claims.csv'
COPYRIGHT_CLAIMS_CSV_F = 'data/plagiarism_dataset/case_data/intermediate_csvs/all_copyright_claims_filtered.csv'
GPT_SONGS_CHECKPOINT_CSV = 'data/plagiarism_dataset/case_data/intermediate_csvs/gpt_songs_checkpoint.csv'
ESTIMATED_SONGS_CSV = 'data/plagiarism_dataset/case_data/intermediate_csvs/estimated_songs.csv'

SONGS_TO_VALIDATE_CSV = 'data/plagiarism_dataset/case_data/intermediate_csvs/songs_to_validate.csv'
VALIDATED_SONGS_CSV = 'data/plagiarism_dataset/case_data/intermediate_csvs/validated_songs.csv'

COPYRIGHT_SONGS_CSV = 'data/plagiarism_dataset/case_data/final_csvs/final_copyright_songs.csv'
COPYRIGHT_PAIRS_CSV = 'data/plagiarism_dataset/case_data/final_csvs/final_copyright_pairs.csv'

# MIDI search log path

MIDI_SEARCH_LOG_PATH = 'data/plagiarism_dataset/midi_search_logs'

# Databases

DATABASE = 'data/copyright_music.db'
SYNTHETIC_DATABASE = 'data/synthetic_music.db'

# MIDI data paths

MIDI_DATA_PATH = 'sms/data/midi_databases'
COPYRIGHT_MIDI_PATH = 'sms/data/copyright_midis'

LAKH_CLEAN_PATH = 'sms/data/midi_databases/lakh_clean'

BIMMUDA_PATH = 'sms/data/midi_databases/bimmuda'
BIMMUDA_METADATA_PATH = 'sms/data/midi_databases/bimmuda/bimmuda_per_song_metadata.csv'

METAMIDI_PATH = 'sms/data/midi_databases/metamidi/MMD_MIDI'
METAMIDI_METADATA_PATH = 'sms/data/midi_databases/metamidi/MMD_scraped_title_artist.jsonl'

MPD_PATH = 'sms/data/midi_databases/MPDSet'

MAESTRO_PATH = 'sms/data/midi_databases/maestro'

# Synthetic data paths

MONOPHONIC_MIDIS_PATH = 'sms/data/synthetic_dataset/monophonic_midis/'
PROCESSED_MELODY_TRACKER = 'sms/data/synthetic_dataset/monophonic_midis/processed_files.csv'

ORIGINAL_SYNTHETIC_MIDIS_PATH = 'sms/data/synthetic_dataset/pairs/original'
PLAGIARISED_SYNTHETIC_MIDIS_PATH = 'sms/data/synthetic_dataset/pairs/plagiarised'
SYNTHETIC_PAIR_TRACKER_PATH = 'sms/data/synthetic_dataset/pairs/synthetic_pairs_tracker.csv'
SYNTHETIC_PAIR_METADATA_PATH = 'sms/data/synthetic_dataset/pairs/synthetic_pair_metadata.pkl'

# MONOPHONIC MIDI PATHS

MAESTRO_PATH = r"C:\Users\cunn2\OneDrive\DSML\Project\thesis-repo\data\synthetic_dataset\monophonic_midis\maestro"
MTC_PATH = r"C:\Users\cunn2\OneDrive\DSML\Project\thesis-repo\data\synthetic_dataset\monophonic_midis\midi_mono"

# Vector databases

MUSIC_VAE_DB = "sms/data/vector_databases/musicvae_embeddings.db"
MUSIC_BERT_DB = "sms/data/vector_databases/musicbert_embeddings.db"

# EXP1

MAESTRO_SEGMENTS_PATH = "data/exp1/maestro_one_bar_segments.pt"
MTC_SEGMENTS_PATH = "data/exp1/mtc_one_bar_segments.pt"
