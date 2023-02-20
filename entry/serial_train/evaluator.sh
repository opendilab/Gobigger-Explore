work_path=$(dirname $0)
export PYTHONPATH="$work_path/../../":$PYTHONPATH
srun --mpi=pmi2 -p cpu -n 1 -c32 --job-name=single_eval python -u -m bigrl.bin.serial_eval  --config $1