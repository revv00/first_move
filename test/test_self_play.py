import sys
import unittest
import logging
from first_move.config import config
from first_move.env.chessboard import ChessBoard
from first_move.env.common import RED
from first_move.alphazero.self_play import play_a_game
from first_move.config import PolicyType

class TestSelfPlay(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.skipIf(True, "skip this test")
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

    @unittest.skipIf(False, "skip this test")
    def test_untrained_model(self):
        config.self_play.red_type = PolicyType('cnn')
        config.self_play.black_type = PolicyType('random')
        config.self_play.red_iter = 0
        config.self_play.black_iter = 0
        hist = play_a_game(iteration=1, config)
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