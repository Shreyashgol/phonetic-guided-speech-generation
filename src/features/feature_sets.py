"""Feature column definitions shared by notebooks 05-08 (and Phase 2 if you reuse the features)."""
CAT_COLS = ['base', 'stress_c', 'cls', 'prev1', 'prev2', 'next1', 'next2', 'word_punct']
NUM_COLS = ['pos_in_word', 'word_len', 'rel_pos_in_word', 'is_word_first', 'is_word_last',
            'word_idx', 'n_words', 'rel_pos_utt', 'seq_idx', 'n_phones', 'is_utt_final', 'is_utt_first',
            'pause_after', 'pause_before', 'is_vowel']
GROUPS = {
    'identity': ['base', 'stress_c', 'cls', 'is_vowel'],
    'context': ['prev1', 'prev2', 'next1', 'next2'],
    'word': ['pos_in_word', 'word_len', 'rel_pos_in_word', 'is_word_first', 'is_word_last'],
    'utterance': ['word_idx', 'n_words', 'rel_pos_utt', 'seq_idx', 'n_phones', 'is_utt_final', 'is_utt_first'],
    'pause_punct': ['pause_after', 'pause_before', 'word_punct'],
}
TARGET = 'dur_f'
