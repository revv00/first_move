from setuptools import setup, find_packages

setup(
    name="first_move",
    version="0.1",
    packages=find_packages(),
    description="Reproduce AlphaZero in Chinese Chess",
    author="Your Name",
    install_requires=[
        "torch==1.10",
        "numpy==1.21",
        "scipy==1.7",
    ],
    entry_points={
        'console_scripts': [
            'self_play = alphazero.self_play:main',
            'train_value_policy = alphazero.train_value_policy:main',
        ]
    }
)
