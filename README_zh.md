# Gobigger-Explore
## :crystal_ball: GoBigger Challenge 2021 Baseline
[en](https://github.com/opendilab/Gobigger-Explore/blob/main/README.md)/[中文](https://github.com/opendilab/Gobigger-Explore/blob/main/README_zh.md)

## :robot: 介绍
这是2021年GoBigger Multi-Agent Decision Intelligence [Challenge](https://www.datafountain.cn/competitions/549)的基线。基线基于[OpenDILab](https://github.com/opendilab/DI-engine)并旨在提供一种简单的入门级方法。参赛选手可以通过扩展提供的基线方法来构建智能体。此外，OpenDILab的模块化结构让参与者可以轻松读懂代码，并且提供了丰富的强化学习算法供参与者使用。对于熟悉多智能体决策AI问题的入门级研究人员来说，这个基线具有较好的指导意义。

## :rocket: Release Version
当前版本为最新的版本v-0.2.0.
1. Version-0.2.0
   - 修复ckpt bug, 提升evaluator评估器的准确性
   - 修复replay_buffer bug。
2. Version-0.1.0
   - 全新的特征工程，提升收敛速度。
   - replay_buffer存放不定长特征，提升数据利用率及训练速度。

## :point_down: 让我们开始吧

1. 实验环境
   - CPU核心数 Core 16
   - GPU显卡 A100(40G)
   - Memory内存 50G
2. 基线参数
   
   - 默认的参数[config](https://github.com/opendilab/Gobigger-Explore/blob/main/my_submission/config/gobigger_no_spatial_config.py)即仓库中开源的参数。参赛选手需根据自己的实验环境配置进行修改。
   - replay_buffer_size的大小需根据系统内存的大小调节。
   - batch_size的大小需要根据显存的大小调节。
   
3. 安装必要的依赖库
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

4. 开始训练
```
    # Download baseline
    git clone https://github.com/opendilab/Gobigger-Explore.git
    cd my_submission/entry/
    python gobigger_vsbot_baseline_main.py
```

5. 评估并保存视频
```
    cd my_submission/entry/
    python gobigger_vsbot_baseline_eval.py --ckpt YOUR_CKPT_PATH
```

## :dart: 实验结果
我们开源了训练的log信息，检查点文件以及评估的视频。链接如下，
   - [百度云盘链接]()
   - [谷歌云盘]()

## :heart_eyes: 相关资源链接
- [比赛主页链接](https://www.datafountain.cn/competitions/549)
- Challenge Repo [github链接](https://github.com/opendilab/GoBigger-Challenge-2021)
- DI-engine Repo [github链接](https://github.com/opendilab/DI-engine)
- GoBigger Repo  [github链接](https://github.com/opendilab/GoBigger)


