# Recognising Four Activities with a Hidden Markov Model Built from Scratch

A Hidden Markov Model (HMM) written in numpy (no HMM library) that tells four activities apart from the accelerometer of one phone: **standing, walking, jumping and still**. Viterbi decoding and Baum-Welch training are both implemented from scratch.

## Why this project

Knowing what a person is doing from a phone is useful for health and wellbeing apps, and most people have only a smartphone, not a wearable. The aim is a low cost activity tracker that runs on one phone at the waist or in the pocket. An HMM fits because activities last for some time and do not change every second, so the order of the windows carries information.

## Main results (unseen test recordings, 360 windows)

| Model | Accuracy |
|---|---|
| Main model (supervised start + Baum-Welch, 23 rounds) | 0.942 |
| Start model (same model, no Baum-Welch) | 1.000 |

Per activity, main model:

| Activity | Windows | Sensitivity | Specificity |
|---|---|---|---|
| standing | 80 | 0.762 | 0.993 |
| walking | 80 | 1.000 | 1.000 |
| jumping | 36 | 1.000 | 1.000 |
| still | 164 | 0.988 | 0.903 |

Things to know before trusting these numbers:

- Baum-Welch raised the training log-likelihood (-1461.36 to -1401.28) but lowered the test accuracy. It does not use the labels and it learned that standing and still swap easily.
- The 1.000 of the start model is partly a test design effect. Every recording has one activity only, so a transition matrix that almost never switches fits the test data very well.
- Added noise breaks the model (see below). It works because the sensor of this phone is clean.
- Only one person and one phone, and 4 test recordings.

## Repository structure

```
README.md
dataset/
  activity_dataset/        the labelled clips (train/ and test/ folders, one folder per activity) and manifest.csv
  Jumping_1-2026-10-23_12-20-22/   example raw recording folder exported by Sensor Logger
notebooks/
  hmm-for-human-activities-data-processing.ipynb   data checks, clips, features, normalisation
  hmm-for-human-activity-states.ipynb              the HMM, training, evaluation, extra experiments
results/
  features.csv                    14 features for each of the 968 windows
  features_normalised.csv         the same, z-scored with the training numbers
  hmm_feature_sets.csv            test accuracy for the different feature sets
  hmm_final_results.csv           final per-activity results, main model and start model
```

- `dataset/activity_dataset/` is the dataset to use. It has 242 clips of 5 seconds at 25 Hz, split into `train/` (152 clips) and `test/` (90 clips) by whole recording, and `manifest.csv`, which lists every clip and the recording it came from.
- The 12 folders next to it are the original recordings. The date and time in a folder name is in UTC, so Kigali time is 2 hours later (for example `Jumping_1-2026-10-23_12-20-22` was recorded at 14:20 in Kigali). The notebook uses the `TotalAcceleration.csv` file in each folder.

## The data

- Sensor Logger app, Samsung SM-A055F (Android), Total Acceleration (x, y, z in m/s², gravity included). The phone has no gyroscope, so only the accelerometer is used.
- Sampling rate: asked for 50 Hz, the phone gave about 27.5 Hz (a sample every 36.3 ms, no long gaps). Every recording was converted to an even 25 Hz grid by linear interpolation.
- 12 separate recordings, each with one activity and each its own start and stop in the app. They were cut into 5 second clips (125 samples) after trimming 1 second from each end. A clip was dropped if it was more than 3 times as shaky as the usual clip of its recording, or (for moving activities) less than 0.4 times as shaky.
- Each clip is a CSV file with the columns `timestamp` (seconds, starts at 0), `acc_x`, `acc_y`, `acc_z` and `activity`. Clip names follow the form `activity_number.csv`; for example, `jumping_001.csv`.
- Train and test are split by whole recording, never by clip, because clips from the same recording look almost the same.

| Activity | Training recordings | Test recording | Clips (train / test) | Files | Total time (s) |
|---|---|---|---|---|---|
| standing | standing1, standing3 | standing2 | 41 / 20 | 61 | 305 |
| walking | walking1 | walking2 | 34 / 20 | 54 | 270 |
| jumping | jumping1, 3, 4, 5 | jumping2 | 41 / 9 | 50 | 250 |
| still | still1 | still2 | 36 / 41 | 77 | 385 |

Unseen data: the four test recordings were made on the evening of 23 October (about 19:08 to 19:20), more than 4 hours after the afternoon training recordings (14:20 to 15:07), and in a different place (the afternoon recordings in the living room, the others in the dining area, which is windier). The person and the phone are the same.

## Features and model

- Windows of 2 seconds (50 samples) with 50% overlap, inside one clip: 968 windows (608 train, 360 test). The window follows from the sampling rate: walking and jumping repeat about 1.6 times per second, so 2 seconds hold about 3 repeats, and the FFT step is 0.5 Hz.
- 14 features per window: 10 time domain (mean x, y, z, log std, log SMA, log range, correlation xy, xz, yz, zero crossing rate) and 4 frequency domain (dominant frequency, log energy in a low and a high band, spectral entropy), from an FFT with a Hann window padded to 256 points.
- Z-score normalisation with the mean and standard deviation of the training windows only.
- Main model features (5, chosen from the F ratios and the correlation check on training data before testing): log_energy_high, zero_cross_rate, dominant_hz, spectral_entropy, corr_xz.
- HMM: 4 hidden states (the activities), diagonal Gaussian emissions, start probabilities and a 4 by 4 transition matrix estimated from the labelled training sequences (one sequence per training recording, 8 in total).
- Everything in log space. Forward, backward, Viterbi, E step and M step are separate documented functions in a `GaussianHMM` class.
- Baum-Welch stops when the change in total log-likelihood is below 1e-4 (200 rounds is only a safety limit). Variance floor 0.05, pseudo count 0.5, seed 42.
- Checks in the notebook: forward and backward agree with brute force, Viterbi equals the brute force path, Viterbi with a flat transition matrix equals the best emission per window, and pure EM never lowers the likelihood on toy data.

## Other experiments (run on the test set for information only, not used to pick the model)

| Experiment | Result |
|---|---|
| Feature sets | all 14: 0.992, motion only (11): 0.872, compact (5, main): 0.942, compact + tilt (8): 1.000 |
| Random start (5 starts) | 2 of 5 reach the same best log-likelihood (-1401.284) and the same 0.942, the other three stay near 0.5 |
| Variance floor 0.05 to 0.001 | trained model falls from 0.942 to 0.828, so a smaller floor does not help |
| Down-sampled to 12.5 Hz | main 0.917, start 0.989 |
| Gaussian noise on the raw clips | from 0.001 m/s² main 0.693 and falling, 0.322 from 0.03 m/s² (every standing and still window decoded as walking) |

The tilt features (mean x, y, z) may be a shortcut, because still has the phone flat and standing has it tilted. The noise result happens because standing and still are almost flat, so their features are tiny energies that a small noise raises to the level of walking.

## How to run it

Both notebooks were run on Kaggle. Run them in this order.

1. `hmm-for-human-activities-data-processing.ipynb`: reads the original recordings in `dataset/`, checks the sampling rate, makes the 25 Hz clips in `dataset/activity_dataset/`, and saves `features.csv` and `features_normalised.csv`. Uses numpy, pandas, matplotlib and scipy.
2. `hmm-for-human-activity-states.ipynb`: add notebook 1 as an input. It loads `features_normalised.csv`, builds and trains the HMM, evaluates it on the test recordings and saves the result tables. The last sections read the clips and `features.csv` from notebook 1. The model itself uses numpy only.

On Kaggle the file paths point to the Kaggle input folders, so to run it elsewhere, change the paths in the settings cell at the top of each notebook.

Every code cell prints `ok` or `FAILED` lines and a final `all checks passed`. The seed is fixed (42), so the numbers above are reproduced on a re-run. The plots (sample signals, feature plots, transition matrix, emissions, confusion matrix, decoded sequences) are shown in the notebooks and in the report.

## Limits

- One person, one phone, 4 test recordings.
- Windows overlap by 50%, so neighbouring windows are not independent.
- Every recording has one activity, so no real switches were learned. The glued test sequence (walking, still, standing, jumping) is made up.
- Walking and still have one training recording each, so no cross-validation by recording was possible.
- No gyroscope.

## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
2. Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Transactions on Information Theory, 13(2), 260-269.
3. Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. The Annals of Mathematical Statistics, 41(1), 164-171.