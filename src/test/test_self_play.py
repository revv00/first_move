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
        hist = play_a_game(config)
        print("History:")
        for brd, turn, policy, value in hist:
            print(f"Turn: {turn}, Value: {value}")
            if turn == RED:
                ChessBoard.print_board(None, brd, indent='')
            else:
                ChessBoard.print_board(None, brd, indent='')
                ChessBoard.print_board(
                    None, ChessBoard.flip_board_and_players(brd), indent='    '
                )

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    unittest.main()