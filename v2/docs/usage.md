# v2 Usage

1) Preprocess + Train (automatically cleans data and trains seq2seq)

    bash v2/scripts/run_train.sh

This reads data/wtbdata_hourly.csv, cleans it and trains a seq2seq LSTM with 24h history -> 24h forecast.

Artifacts saved to:
- v2/results/cleaned.csv
- v2/results/artifacts/feature_scaler.joblib
- v2/results/artifacts/target_scaler.joblib
- v2/results/checkpoints/seq2seq/best.pth and last.pth

2) Evaluate

    bash v2/scripts/run_eval.sh

Saves metrics to v2/results/metrics_seq2seq.json (MAE, RMSE).

3) Predict + Scenario Sampling

    bash v2/scripts/run_predict.sh

Generates:
- v2/results/seq2seq_y_true.npy
- v2/results/seq2seq_y_pred_mean.npy
- v2/results/seq2seq_y_samples_first.npy

4) Visualize Scenarios

    bash v2/scripts/run_vis.sh

Saves plot to v2/results/plots/seq2seq_scenarios.png.

Notes
- Cleaning rules:
  - Remove rows with power == 0
  - Remove rows with wind speed == 0
  - Remove rows with wind speed <= (power / 50)
- Splits are per turbine id: 8:1:1 along the time axis.
- Only continuous hourly sequences are used to build windows.
- MC Dropout is used at inference to generate 100 scenario samples.