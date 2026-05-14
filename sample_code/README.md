# anna parilova 2025

сodebase for my Bachelor thesis experiments

- percom paper https://www.overleaf.com/8396193725prctcwqwdnwz#a4fa92
- percom presentation https://www.overleaf.com/9321943849fknmdnwggvqx#3e095e
- thesis https://www.overleaf.com/6184246637jqzbhwhghtmn#2846d5 

## how to train and save best models
### 0. setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

unzip archives from `zip_data/` into `data/` so paths match scripts:

- IMU: `data/imu/Uxx/...`
- infrared: `data/infrared/*.csv`
- radar: `data/radar/Uxx/...`

### 1. preprocess with best parameters

reads `best_models_config.toml` and runs preprocessing for the selected best configs:

- `IMU`: merged `Chest + RightArm + LeftArm`, `6s`, `75%`, `FEAT`
- `INFRARED`: `E03 E04`, `12s`, `25%`, `FEAT`
- `RADAR`: `E03 E04`, `8s`, `75%`, `FEAT`

```bash
python3 src/best/preprocess_best.py
```

you can also select data and events explicitly:

```bash
python src/best/preprocess_best.py --data IMU INFRARED RADAR
python src/best/preprocess_best.py --data IMU --events E01 E02 E03 E04
```

generated feature files will be saved into:

- `processed/imu/feats/`
- `processed/infrared/feats/`
- `processed/radar/feats/`

### 2. train and save best models

reads `best_models_config.toml` and trains the final selected classical models on the full processed feature sets

```bash
python src/best/train_best_models.py
```

to train models on a validation split instead of the full dataset:

```bash
python src/best/train_best_models.py \
  --data IMU INFRARED RADAR \
  --use-validation True \
  --val-size 0.2
```

saved model artifacts:

- `models/imu_svm_best.pkl`
- `models/infrared_rf_best.pkl`
- `models/radar_svm_best.pkl`

if `--use-validation True`, files are saved with `_val` suffix:

- `models/imu_svm_best_val.pkl`
- `models/infrared_rf_best_val.pkl`
- `models/radar_svm_best_val.pkl`

each `.pkl` bundle contains:

- trained sklearn model
- fitted scaler
- fitted label encoder
- metadata with config and metrics
- if `--use-validation True`, also the saved validation split indices

## how to use saved models

`predict_best.py` runs prediction with already saved best models

by default it can:
- optionally run best preprocessing first
- load saved model bundles from `models/`
- load matching `FEAT` files from `processed/.../feats/`
- save predictions into timestamped folders inside `predictions/`

```bash
python src/best/predict_best.py
```

predict only on the saved validation split:

```bash
python src/best/predict_best.py \
  --data IMU INFRARED RADAR \
  --preprocess False \
  --use-validation True
```

predict for a different event set:

```bash
python src/best/predict_best.py \
  --data IMU INFRARED RADAR \
  --events E01 E02 E03 E04 \
  --preprocess False
```

### output structure

prediction files are saved like this:

- `predictions/<modality>/e03_e04/<timestamp>/best_predictions.csv`
- `predictions/<modality>/e03_e04/<timestamp>/best_user_predictions.csv`

run summary is saved into:

- `predictions/e03_e04/<timestamp>/best_prediction_summary.json`

`best_predictions.csv` contains per-window predictions:

- `index`
- `pred_label`
- `pred_score`
- `true_label`
- `user`

`best_user_predictions.csv` contains majority-vote predictions per user:

- `user`
- `true_label`
- `pred_label`
- `correct`

## how to reproduce experiments
### 0. setup (same steps as above)

### 1. preprocessing

#### imu
```bash
python src/preprocessing/imu_preprocessing.py --run everything
```

options:
- `--run raw|resample|segment|transform|everything`
- `--durations 4 6 8 12`
- `--overlaps 0 25 50 75`
- `--method RP|GAF|SPEC|MTF|LINE|FEAT`
- `--sensors Chest RightArm LeftArm`
- `--merge-sensors True`

#### infrared
```bash
python src/preprocessing/infrared_preprocessing.py --run everything --events E03 E04
```

#### radar
```bash
python src/preprocessing/radar_preprocessing.py --run everything --events E03 E04
```

### 2. run models

#### deep models (cnn/rnn)
```bash
python src/train/run_models.py \
  --data INFRARED \
  --model CNN \
  --louo True \
  --durations 6 \
  --overlaps 0 \
  --reprs RP \
  --search random \
  --trials 20 \
  --out-csv res/infrared_cnn_run/run_louo.csv \
  --out-best res/infrared_cnn_run/run_louo_best.json
```

#### classical models
```bash
python src/train/run_classifiers.py \
  --data INFRARED \
  --louo True \
  --durations 6 \
  --overlaps 0 \
  --classifiers logreg svm rf \
  --search random \
  --trials 20 \
  --out-csv res/infrared_classical_run/run_louo.csv \
  --out-best res/infrared_classical_run/run_louo_best.json
```

run outputs include:
- `acc` - balanced mean accuracy per window
- `window_acc`- mean accuracy per window
- `louo_acc_std`- std for window accuracy for each user if running leave-one-user-out
- `user_acc`- if we guessed more than a half user windows correcly, than we guessed the user emotion
- `combined_acc`- harmonic average between acc and user_acc
- `user_report_json`- confusion matrix for guessing per user
- `per_user_window_acc_json` - window accuracy for each user if running leave-one-user-out

#### 3. segmentation experiments (script for slurm jobs)

main runner:
```bash
bash jobs/run_experiments_sequential.sh jobs/experiments.csv
```

you can also pass a custom CSV (for example one modality only):
```bash
bash jobs/run_experiments_sequential.sh jobs/imu_cnn.csv
```

notes:
- i provided examples in examples folder
- if `duration=multi` or `overlap=multi`, runner expands to all 16 pairs
- results are checkpointed per pair
