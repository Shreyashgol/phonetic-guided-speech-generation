# Phase 2 summary (auto-generated)

Acoustic model: 8.84 M parameters, best checkpoint at step 8000, best validation mel L1 0.4073 (normalised units).

## Objective duration-swap results (test)

| condition     |   dtw_l1 |   ci_lo |   ci_hi |   total_len_err_frames |   len_ci_lo |   len_ci_hi |
|:--------------|---------:|--------:|--------:|-----------------------:|------------:|------------:|
| ground_truth  |   0.6617 |  0.6562 |  0.6674 |                  0     |      0      |      0      |
| phase1_bilstm |   0.6646 |  0.6592 |  0.6702 |                 20.15  |     17.5998 |     22.6751 |
| phone_mean    |   0.6782 |  0.6727 |  0.6837 |                 48.45  |     43.575  |     53.4701 |
| global_mean   |   0.6859 |  0.6808 |  0.6914 |                 42.25  |     37.6699 |     47.1006 |
| own_head      |   0.6661 |  0.6606 |  0.6715 |                 19.855 |     17.3249 |     22.435  |

## Paired differences

| comparison                   |   mean_diff |   ci_lo |   ci_hi |   share_a_better |
|:-----------------------------|------------:|--------:|--------:|-----------------:|
| phase1_bilstm - phone_mean   |     -0.0136 | -0.0155 | -0.0116 |            0.895 |
| phase1_bilstm - global_mean  |     -0.0213 | -0.0243 | -0.0185 |            0.92  |
| phase1_bilstm - own_head     |     -0.0015 | -0.0023 | -0.0005 |            0.57  |
| phase1_bilstm - ground_truth |      0.0029 |  0.0015 |  0.0043 |            0.375 |
| phone_mean - ground_truth    |      0.0165 |  0.0142 |  0.0187 |            0.14  |

## Listening test (MOS)

| condition       |   mos |   ci_lo |   ci_hi |
|:----------------|------:|--------:|--------:|
| ground_truth    |  4.17 |    3.97 |    4.33 |
| natural_vocoded |  4.27 |    4.13 |    4.43 |
| phase1_bilstm   |  4.13 |    3.93 |    4.33 |
| phone_mean      |  4.37 |    4.17 |    4.53 |

## Vocoder copy-synthesis

| utt_id     |   seconds |   mel_l1_vs_original |
|:-----------|----------:|---------------------:|
| LJ001-0005 |      8.22 |              1.27814 |
| LJ001-0065 |      7.19 |              1.29074 |
| LJ001-0099 |      8.14 |              1.23843 |
| LJ001-0155 |      4.37 |              1.33739 |
| LJ001-0162 |      9.47 |              1.3395  |