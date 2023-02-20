from setuptools import setup

setup(
    name='bigger_rl',
    version='0.0.1',
    description='bigger_rl - bot for game Game Of Art of War',
    author='X-lab',
    license='Apache License, Version 2.0',
    keywords='game AI',
    packages=[
        'game_zoo',
        'bigrl',
        'bigrl.core'
    ],
    install_requires=[
        'psutil',
        # 'grpcio',
        'easydict',
        'matplotlib',
        # 'google-api-python-client',
        'pyyaml',
        # 'opencv-python',
        'tabulate',
        'tensorboardX',
        # 'protobuf3',
        # 'pyarrow',
        'lz4',
        'redis',
        'portpicker',
        'tensorboard',
        # 'redis',
        'flask',
        # 'grpcio',
        # 'protobuf',
    ]
)
