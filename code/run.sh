joinByChar() {
  local IFS="$1"
  shift
  echo "$*"
}

python -u train_v2.py
python -u train_v3.py

results=($(ls result/*.csv))
results=`joinByChar ' ' "${results[@]}"`
python -u gen_final_result.py --files $results
