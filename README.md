# First Move: Reproduce AlphaZero in Chinese Chess

## Background

It has been years since the success of the AlphaGo series of algorithms, and this project is yet another reproduction of the algorithm applied to Chinese Chess. This repository is greatly inspired by [NeymarL's ChineseChess-AlphaZero](https://github.com/NeymarL/ChineseChess-AlphaZero).

The motivation to "reinvent the wheel" is twofold: firstly, it stems from curiosity; secondly, I have some intuitive ideas that I want to test. Additionally, with PyTorch now the de facto standard for deep learning algorithms, this project aims to leverage its flexibility and power to explore improvements in learning efficiency.

## Code Structure

The project is organized as follows:

```
first_move/
├── alphazero/
│   ├── self_play.py           # Facilitates self-play, starting from scratch or a checkpoint
│   ├── train_value_policy.py  # Trains the value and policy networks
│
├── env/
│   ├── board.py               # Contains board-related logic and rules for Chinese Chess
│
├── model/
│   ├── model.py               # Defines the neural network model (plane input, handles turns)
│   ├── model_service.py       # Provides services related to model operations
```

## Installation

To install this repository in development mode, run:

```bash
python setup.py develop
```
