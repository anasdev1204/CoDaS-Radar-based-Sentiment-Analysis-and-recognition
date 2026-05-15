#!/bin/bash
set -euo pipefail

CSV_PATH="${1:-jobs/experiments.csv}"
JOBS_DIR="jobs/sequential_jobs"
LOGS_DIR="logs/sequential"
RESULTS_DIR="res/segmentation_experiments"
PAIRS_DIR="${RESULTS_DIR}/pairs"
MERGED_CSV="${RESULTS_DIR}/merged_results.csv"

DEFAULT_DURATIONS=(4 6 8 12)
DEFAULT_OVERLAPS=(0 25 50 75)

RP_IMG_SIZE="${RP_IMG_SIZE:-600}"
RESET_RESULTS="${RESET_RESULTS:-0}"
RESUME="${RESUME:-1}"

mkdir -p "$JOBS_DIR" "$LOGS_DIR" "$RESULTS_DIR" "$PAIRS_DIR"

if [ ! -f "$CSV_PATH" ]; then
  echo "CSV not found: $CSV_PATH"
  exit 1
fi

if [ ! -f "$MERGED_CSV" ]; then
  cat > "$MERGED_CSV" <<'EOF'
exp_id,data,model,duration,overlap,acc,user_acc,combined_acc,pair_csv,pair_best,status,timestamp
EOF
fi

submit_job() {
  local job_file="$1"
  local dep="$2"
  local out
  if [ -n "$dep" ]; then
    out=$(sbatch --dependency=afterok:"$dep" "$job_file")
  else
    out=$(sbatch "$job_file")
  fi
  echo "$out" | awk '{print $4}'
}

prev_job_id=""

while IFS=, read -r \
  exp_id data model duration overlap sensor_tag epochs batch_size lr dropout \
  cnn_arch conv1 conv2 fc_hidden rnn_arch hidden_dim num_layers clf_arch \
  logreg_c logreg_solver logreg_max_iter svm_c svm_kernel svm_gamma \
  rf_n_estimators rf_max_depth rf_min_samples_split rf_min_samples_leaf
do
  if [ "$exp_id" = "exp_id" ] || [ -z "$exp_id" ]; then
    continue
  fi

  data_lc=$(echo "$data" | tr '[:upper:]' '[:lower:]')
  if [ "$data" = "IMU" ]; then
    sensor_arg="--sensor-tag IMU"
  else
    sensor_arg="--sensor-tag NA"
  fi

  exp_results_csv="${RESULTS_DIR}/${exp_id}_results.csv"
  exp_best_json="${RESULTS_DIR}/${exp_id}_best.json"
  done_pairs_file="${RESULTS_DIR}/${exp_id}_done_pairs.txt"
  exp_pairs_dir="${PAIRS_DIR}/${exp_id}"
  mkdir -p "$exp_pairs_dir"

  if [ "$RESET_RESULTS" = "1" ]; then
    rm -f "$exp_results_csv" "$exp_best_json" "$done_pairs_file"
    rm -f "${exp_pairs_dir}"/*.csv "${exp_pairs_dir}"/*.json 2>/dev/null || true
  fi
  touch "$done_pairs_file"

  pair_specs=()
  if [ "$duration" = "multi" ] || [ "$overlap" = "multi" ]; then
    for d in "${DEFAULT_DURATIONS[@]}"; do
      for o in "${DEFAULT_OVERLAPS[@]}"; do
        pair_specs+=("${d}:${o}")
      done
    done
  else
    pair_specs+=("${duration}:${overlap}")
  fi

  for pair_spec in "${pair_specs[@]}"; do
      d="${pair_spec%%:*}"
      o="${pair_spec##*:}"
      job_name="${exp_id}_d${d}_o${o}"
      job_file="${JOBS_DIR}/${job_name}.slurm"
      pair_csv="${exp_pairs_dir}/${job_name}.csv"
      pair_best="${exp_pairs_dir}/${job_name}_best.json"

      cat > "$job_file" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOGS_DIR}/${job_name}_%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1

set -euo pipefail
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] start ${job_name}"

PROJECT_ROOT="\${SLURM_SUBMIT_DIR:-\$(cd "\$(dirname "\$0")/../.." && pwd)}"
cd "\$PROJECT_ROOT"

module load triton/2025.1-gcc || true
module load python/3.11.9 || true

if [ -f /scratch/work/pariloa1/thesis_env/bin/activate ]; then
  source /scratch/work/pariloa1/thesis_env/bin/activate
elif [ -f thesis_env/bin/activate ]; then
  source thesis_env/bin/activate
elif [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

export PYTHONUNBUFFERED=1
RP_IMG_SIZE="${RP_IMG_SIZE}"
RESUME="${RESUME}"

if [ "\$RESUME" = "1" ] && grep -qx "${d},${o}" "${done_pairs_file}"; then
  echo "[\$(date '+%Y-%m-%d %H:%M:%S')] skip already done pair d=${d} o=${o}"
  exit 0
fi

cleanup_pair() {
  rm -f processed/${data_lc}/windows/*win${d}s_overlap${o}*.npy || true
  rm -f processed/${data_lc}/RP/*win${d}s_overlap${o}*.npy || true
  rm -f processed/${data_lc}/feats/*win${d}s_overlap${o}*.npy || true
}
trap cleanup_pair EXIT

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] pair start d=${d} o=${o}"
EOF

      if [ "$data" = "IMU" ]; then
        cat >> "$job_file" <<EOF
python src/preprocessing/imu_preprocessing.py --run segment --durations ${d} --overlaps ${o} --sensors Chest RightArm LeftArm --merge-sensors True
EOF
      elif [ "$data" = "INFRARED" ]; then
        cat >> "$job_file" <<EOF
python src/preprocessing/infrared_preprocessing.py --run segment --durations ${d} --overlaps ${o}
EOF
      else
        cat >> "$job_file" <<EOF
python src/preprocessing/radar_preprocessing.py --run segment --durations ${d} --overlaps ${o} --events E03 E04
EOF
      fi

      if [ "$model" = "CNN" ]; then
        if [ "$data" = "IMU" ]; then
          cat >> "$job_file" <<EOF
python src/preprocessing/imu_preprocessing.py --run transform --method RP --durations ${d} --overlaps ${o} --resize True --image-size \$RP_IMG_SIZE --sensors Chest RightArm LeftArm --merge-sensors True
EOF
        elif [ "$data" = "INFRARED" ]; then
          cat >> "$job_file" <<EOF
python src/preprocessing/infrared_preprocessing.py --run transform --method RP --durations ${d} --overlaps ${o} --resize True --image-size \$RP_IMG_SIZE
EOF
        else
          cat >> "$job_file" <<EOF
python src/preprocessing/radar_preprocessing.py --run transform --method RP --tag win${d}s_overlap${o} --resize True --image-size \$RP_IMG_SIZE
EOF
        fi
      elif [ "$model" = "CLASSICAL" ]; then
        if [ "$data" = "IMU" ]; then
          cat >> "$job_file" <<EOF
python src/preprocessing/imu_preprocessing.py --run transform --method FEAT --durations ${d} --overlaps ${o} --resize False --sensors Chest RightArm LeftArm --merge-sensors True
EOF
        elif [ "$data" = "INFRARED" ]; then
          cat >> "$job_file" <<EOF
python src/preprocessing/infrared_preprocessing.py --run transform --method FEAT --durations ${d} --overlaps ${o} --resize False
EOF
        else
          cat >> "$job_file" <<EOF
python src/preprocessing/radar_preprocessing.py --run transform --method FEAT --tag win${d}s_overlap${o} --resize False
EOF
        fi
      fi

      if [ "$model" = "CNN" ]; then
        cat >> "$job_file" <<EOF
python src/training/run_models.py --data ${data} --model CNN --louo True ${sensor_arg} --durations ${d} --overlaps ${o} --reprs RP --img-size \$RP_IMG_SIZE --search grid --trials 1 --archs ${cnn_arch} --epochs ${epochs} --batch-sizes ${batch_size} --lrs ${lr} --dropouts ${dropout} --conv1 ${conv1} --conv2 ${conv2} --fc-hidden ${fc_hidden} --best-by combined_acc --out-csv "${pair_csv}" --out-best "${pair_best}"
EOF
      elif [ "$model" = "RNN" ]; then
        cat >> "$job_file" <<EOF
python src/training/run_models.py --data ${data} --model RNN --louo True ${sensor_arg} --durations ${d} --overlaps ${o} --search grid --trials 1 --rnn-archs ${rnn_arch} --epochs ${epochs} --batch-sizes ${batch_size} --lrs ${lr} --dropouts ${dropout} --hidden-dims ${hidden_dim} --num-layers ${num_layers} --best-by combined_acc --out-csv "${pair_csv}" --out-best "${pair_best}"
EOF
      else
        cat >> "$job_file" <<EOF
python src/training/run_classifiers.py --data ${data} --louo True ${sensor_arg} --durations ${d} --overlaps ${o} --classifiers ${clf_arch} --search grid --trials 1 --best-by combined_acc --logreg-c ${logreg_c:-1.0} --logreg-solver ${logreg_solver:-lbfgs} --logreg-max-iter ${logreg_max_iter:-1000} --svm-c ${svm_c:-1.0} --svm-kernel ${svm_kernel:-rbf} --svm-gamma ${svm_gamma:-scale} --rf-n-estimators ${rf_n_estimators:-100} --rf-max-depth ${rf_max_depth:-0} --rf-min-samples-split ${rf_min_samples_split:-2} --rf-min-samples-leaf ${rf_min_samples_leaf:-1} --out-csv "${pair_csv}" --out-best "${pair_best}"
EOF
      fi

      cat >> "$job_file" <<EOF
python - <<'PY'
from pathlib import Path
import pandas as pd

pair_csv = Path("${pair_csv}")
exp_csv = Path("${exp_results_csv}")
if pair_csv.exists() and pair_csv.stat().st_size > 0:
    df = pd.read_csv(pair_csv)
    if exp_csv.exists() and exp_csv.stat().st_size > 0:
        df.to_csv(exp_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(exp_csv, index=False)
PY

echo "${d},${o}" >> "${done_pairs_file}"

python - <<'PY'
import csv
import json
import os
import time
from pathlib import Path
import pandas as pd

exp_csv = Path("${exp_results_csv}")
exp_best = Path("${exp_best_json}")
if exp_csv.exists() and exp_csv.stat().st_size > 0:
    df = pd.read_csv(exp_csv)
    best_col = "combined_acc" if "combined_acc" in df.columns else "acc"
    valid = df.dropna(subset=[best_col]) if best_col in df.columns else pd.DataFrame()
    if len(valid) > 0:
        best = valid.sort_values(best_col, ascending=False).iloc[0].to_dict()
        exp_best.write_text(json.dumps(best, indent=2), encoding="utf-8")

pair_best_path = "${pair_best}"
row = {
    "exp_id": "${exp_id}",
    "data": "${data}",
    "model": "${model}",
    "duration": "${d}",
    "overlap": "${o}",
    "acc": "",
    "user_acc": "",
    "combined_acc": "",
    "pair_csv": "${pair_csv}",
    "pair_best": pair_best_path,
    "status": "failed",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
if os.path.exists(pair_best_path):
    with open(pair_best_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    if "error" not in j:
        row["acc"] = j.get("acc", "")
        row["user_acc"] = j.get("user_acc", "")
        row["combined_acc"] = j.get("combined_acc", "")
        row["status"] = "ok"

with open("${MERGED_CSV}", "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "exp_id","data","model","duration","overlap",
            "acc","user_acc","combined_acc","pair_csv","pair_best","status","timestamp"
        ],
    )
    w.writerow(row)
PY

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] done ${job_name}"
EOF

      chmod +x "$job_file"
      job_id=$(submit_job "$job_file" "$prev_job_id")
      echo "submitted ${job_name}: ${job_id} (afterok=${prev_job_id:-none})"
      prev_job_id="$job_id"
  done
done < "$CSV_PATH"

echo "all pair jobs submitted sequentially"
echo "last job id: ${prev_job_id}"
echo "track with: squeue -u \$USER"
