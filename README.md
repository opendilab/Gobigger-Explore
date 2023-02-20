## Installation

You can simply install bigger_rl from PyPI with the following command:
```bash
git clone git/repo/path
cd bigger_rl
pip install -e .
```

## QuickStart: Example to train cartpole in local device
The following is an example to train cartpole


1. Serial Training(DQN、PPO、MAPPO、QMix、VMix、COMA)
- Train
    ```
    python -u -m bigrl.bin.serial_train  --config abs/config/path
    ```
- Evaluation
    ```
    python -u -m bigrl.bin.serial_eval  --config abs/config/path
    ```


2. Parallel Training(IMPALA + Alphastar_Loss + League Training)
- Train && Evaluation
    ```
    python -u -m bigrl.single.worker.trainer.trainer  --config abs/config/path
    ```