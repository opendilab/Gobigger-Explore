# Gobigger-Explore
## :crystal_ball: GoBigger Challenge 2021 Baseline
[en](https://github.com/opendilab/Gobigger-Explore/blob/main/README.md)/[中文](https://github.com/opendilab/Gobigger-Explore/blob/main/README_zh.md)

## :robot: Introduction
This is the baseline of GoBigger Multi-Agent Decision Intelligence [Challenge](https://www.datafountain.cn/competitions/549) in 2021. The baseline is based on [OpenDILab](https://github.com/opendilab/DI-engine) and aims to provide a simple entry-level method. Participants can build agents by extending the baseline method provided. In addition, Opendilab's modular structure allows participants to easily get started, and it provides a wealth of reinforcement learning algorithms for participants to use. This baseline is a good starting point, especially for entry-level researchers who are familiar with multi-agent decision AI problems.

## :rocket: Release Version
The current version is the latest version v-0.2.0.
1. Version-0.2.0
   - Fix the ckpt bug to improve the accuracy of the evaluator.
   - Fix replay_buffer bug.
2. Version-0.1.0
   - Brand new feature engineering to improve convergence speed.
   - Replay_buffer stores variable-length features to improve data utilization and training speed.

## :point_down: Getting Started

1. System environment
   - Core 16
   - GPU A100(40G)
   - Memory 50G
2. Baseline Config
   
   - The default config is the [config](https://github.com/opendilab/Gobigger-Explore/blob/main/my_submission/config/gobigger_no_spatial_config.py) used in this experiment. Participants can modify it according to the system environment.
   - The size of replay_buffer_size needs to be set according to the size of RAM.
   - The size of batch_size needs to be set according to the size of the GPU memory.
   
3. Install the necessary packege
```
    # Install DI-engine
    git clone https://github.com/opendilab/DI-engine.git
    cd YOUR_PATH/DI-engine/
    pip install -e . --user

    # Install Env Gobigger
    git clone https://github.com/opendilab/GoBigger.git
    cd YOUR_PATH/GoBigger/
    pip install -e . --user
```

4. Start training
```
    # Download baseline
    git clone https://github.com/opendilab/Gobigger-Explore.git
    cd my_submission/entry/
    python gobigger_vsbot_baseline_main.py
```

5. Evaluator and Save game videos
```
    cd my_submission/entry/
    python gobigger_vsbot_baseline_eval.py --ckpt YOUR_CKPT_PATH
```

## :dart: Result
We released training log information, checkpoints, and evaluation videos. Below is the download link,
   - Baidu Netdisk
   - Google Drive


## :heart_eyes: Resources
- [Challenge Page Link](https://www.datafountain.cn/competitions/549)
- Challenge Repo [Github Link](https://github.com/opendilab/GoBigger-Challenge-2021)
- DI-engine Repo [Github Link](https://github.com/opendilab/DI-engine)
- GoBigger Repo [Github Link](https://github.com/opendilab/GoBigger)


