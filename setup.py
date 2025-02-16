from setuptools import setup, find_packages

setup(
    name="first_move",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="Reproduce AlphaZero in Chinese Chess",
    author="revv",
    install_requires=[
        "torch==2.2.2",
        "torchvision==0.17.2",
        "numpy==1.26.4",
        "scipy==1.13.0",
        "colorama==0.4.6",
        "pandas==2.1.4",
        "pyarrow==16.1.0",
        "pyzmq==26.2.1",
    ],
    entry_points={
        'console_scripts': [
            'self_play = alphazero.self_play:main',
            'train_value_policy = alphazero.train_value_policy:main',
        ]
    }
)
