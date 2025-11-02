# v2 Architecture

- src/preprocess.py
  - load_and_clean(): robust column detection, three cleaning filters, adds hour/dayofweek.
- src/dataset.py
  - WindSeqIndexer: builds continuous segments per turbine and sliding window indices.
  - Split 8:1:1 per turbine by time.
  - WindSeqDataset: returns standardized inputs, scaled targets for training.
- src/model_seq2seq.py
  - LSTM Encoder-Decoder. Decoder uses previous output and supports teacher forcing.
  - Dropout enabled for MC sampling during inference.
- src/train.py
  - Trains with MSE on scaled targets, ReduceLROnPlateau, early stopping, gradient clipping.
  - Saves best/last checkpoints and scalers.
- src/evaluate.py
  - Loads best checkpoint, inverts scaling, reports MAE and RMSE.
- src/predict.py
  - MC Dropout to sample 100 scenarios; saves arrays for visualization.
- src/visualize.py
  - Plots true vs mean prediction and spaghetti of 100 scenarios.

Data Flow
raw CSV -> preprocess (cleaned.csv) -> index/split -> DataLoader -> train/evaluate -> predict -> visualize.