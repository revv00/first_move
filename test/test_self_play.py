import sys
import unittest
import logging
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
        last_brd = None
        for brd, turn, policy, value in hist:
            print(f"Turn: {turn}, Value: {value}")
            if turn == RED:
                ChessBoard.print_board(last_brd, brd, indent='', level=logging.ERROR)
            else:
                ChessBoard.print_board(last_brd, brd, indent='', level=logging.ERROR)
                """
                ChessBoard.print_board(
                    ChessBoard.flip_board_and_players(last_brd),
                    ChessBoard.flip_board_and_players(brd), indent='    ', level=logging.ERROR
                )
                """
                
            last_brd = brd

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    unittest.main()