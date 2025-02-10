import sys
import unittest
import torch
import numpy as np
from config import config
from env.chessboard import ChessBoard
from env.common import RED
from model.value_policy_net import ValuePolicyNet

class TestModel(unittest.TestCase):

    def setUp(self):
        self.board = ChessBoard()

    def test_untrained_model(self):
        # ulimit -n 65536 to increate file opening count
        data_root = 'data/vboards/soldier_king1.txt'
        print("Test read board from file")
        board = ChessBoard.read_visualization_from_file(data_root)
        cb = ChessBoard()
        cb.assign_board(board, turn=RED)

        plane = torch.tensor(cb.get_plane()[np.newaxis, :])
        model = ValuePolicyNet(config=config.model)
        policy, value = model(plane)
        print(policy, value)

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    unittest.main()