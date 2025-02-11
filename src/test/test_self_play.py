import sys
import unittest
import torch
import numpy as np
from config import config
from env.chessboard import ChessBoard
from env.common import RED
from alphazero.self_play import play_a_game

class TestSelfPlay(unittest.TestCase):

    def setUp(self):
        pass

    def test_untrained_model(self):
        play_a_game(config)

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    unittest.main()