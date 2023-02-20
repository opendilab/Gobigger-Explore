work_path=$(dirname $0)
export PYTHONPATH="$work_path/../../":$PYTHONPATH
srun --mpi=pmi2 -p GAME -N 1 --gres=gpu:1 --ntasks-per-node 1 -c 32 --job-name=serial_train  python -u -m bigrl.bin.serial_train  --config $1