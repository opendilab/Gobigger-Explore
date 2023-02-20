work_path=$(dirname $0)
export PYTHONPATH="$work_path/../../":$PYTHONPATH
python -u -m bigrl.bin.serial_eval  --config $1