import unittest
import copy

from env.chessboard import ChessBoard
from env.common import RED

class TestChessBoard(unittest.TestCase):

    def setUp(self):
        self.board = ChessBoard()

    def test_initial_setup(self):
        ChessBoard.print_board(None, self.board.board)
        # move 
        prev_board = copy.deepcopy(self.board.board)
        self.board.move_action_str("0304")
        ChessBoard.print_board(prev_board, self.board.board)

    def test_read_from_visualization(self):
        visualization = \
"""    a    b    c    d    e    f    g    h    i
 +----+----+----+----+----+----+----+----+----+
9| 車 | 馬 | 象 | 士 | 將 | 士 | 象 | 馬 | 車 |
 +----+----+----+----+----+----+----+----+----+
8| .. | .. | .. | .. | .. | .. | .. | .. | .. |
 +----+----+----+----+----+----+----+----+----+
7| .. | 砲 | .. | .. | .. | .. | .. | 砲 | .. |
 +----+----+----+----+----+----+----+----+----+
6| 卒 | .. | 卒 | .. | 卒 | .. | 卒 | .. | 卒 |
 +----+----+----+----+----+----+----+----+----+
5| .. | .. | .. | .. | .. | .. | .. | .. | .. |
 +----+----+----+----+----+----+----+----+----+
4| 兵 | .. | .. | .. | .. | .. | .. | .. | .. |
 +----+----+----+----+----+----+----+----+----+
3| .. | .. | 兵 | .. | 兵 | .. | 兵 | .. | 兵 |
 +----+----+----+----+----+----+----+----+----+
2| .. | 炮 | .. | .. | .. | .. | .. | 炮 | .. |
 +----+----+----+----+----+----+----+----+----+
1| .. | .. | .. | .. | .. | .. | .. | .. | .. |
 +----+----+----+----+----+----+----+----+----+
0| 俥 | 傌 | 相 | 仕 | 帥 | 仕 | 相 | 傌 | 俥 |
 +----+----+----+----+----+----+----+----+----+
    a    b    c    d    e    f    g    h    i"""
        print("Test read board from string")
        
        # Parse the visualization
        board = ChessBoard.parse_visualization(visualization)

        # Print the internal representation
        for row in board:
            print(row)

    def test_read_from_file_and_move(self):
        data_root = 'data/vboards/soldier_king1.txt'
        print("Test read board from file")
        board = ChessBoard.read_visualization_from_file(data_root)
        cb = ChessBoard()
        cb.assign_board(board, turn=RED)
        
        # Print the internal representation
        #for row in board:
        #    print(row)
        prev_board = copy.deepcopy(board)
        cb.move_action_str("5051")
        ChessBoard.print_board(prev_board, cb.board)

        # Force it to be RED
        prev_board = copy.deepcopy(cb.board)
        cb.assign_board(prev_board, turn=RED)
        cb.move_action_str("6858")
        ChessBoard.print_board(prev_board, cb.board)

        # Force it to be RED again， should now be end of game
        prev_board = copy.deepcopy(cb.board)
        cb.assign_board(prev_board, turn=RED)
        cb.move_action_str("5848")
        ChessBoard.print_board(prev_board, cb.board)
        print(cb.is_end())
        

    def test_move_piece(self):
        pass

if __name__ == '__main__':
    unittest.main()