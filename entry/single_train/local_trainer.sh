work_path=$(dirname $0)
export PYTHONPATH="$work_path/../../":$PYTHONPATH
python -u -m bigrl.single.worker.trainer.trainer  --config $1