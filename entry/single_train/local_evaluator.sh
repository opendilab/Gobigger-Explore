work_path=$(dirname $0)
export PYTHONPATH="$work_path/../../":$PYTHONPATH
python -u -m bigrl.single.worker.evaluator.evaluator  --config $1