import sys
import unittest
import copy
from env.chessboard import ChessBoard
from env.common import RED
from alphazero.mcts import MCTS
from config import config

class TestChessBoard(unittest.TestCase):

    def setUp(self):
        self.board = ChessBoard()

    def test_read_and_mcst_once(self):
        # ulimit -n 65536 to increate file opening count
        data_root = 'data/vboards/soldier_king1.txt'
        print("Test read board from file")
        board = ChessBoard.read_visualization_from_file(data_root)
        cb = ChessBoard()
        cb.assign_board(board, turn=RED)
        
        mcst = MCTS(config.self_play)
        # sequential calling srch_once
        # NOTE: might be infinite loop
        for i in range(2000):
            mcst.mcts_srch_once(copy.deepcopy(cb), RED)

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    unittest.main()