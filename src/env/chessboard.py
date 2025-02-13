#! /usr/bin/env python
# -*- coding: utf-8 -*-

# pycchess - just another chinese chess UI
# Copyright (C) 2011 - 2015 timebug

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# Shamelessly copied from https://github.com/NeymarL/ChineseChess-AlphaZero
import hashlib
import logging
from .common import *
from enum import Enum
import numpy as np
from colorama import Fore, Back, Style, init
init(strip=False, autoreset=True) # strip for dumping control characters

# Create a specific logger for this function
logger = logging.getLogger('board_logger')

# Set up a custom logger for this function (will not affect the global logger)
logger.setLevel(logging.INFO)
logger.setLevel(logging.ERROR)
# Create a StreamHandler to output to the console
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

Winner = Enum("Winner", "red black draw")

class ChessBoard:
    """
    Chessboard for Chinese Chess. board is a 10x9 2D array, following natrual cartesian coordinate system.
    This chessboard is designed for holding state for normal 1v1 play. If you need to do MCTS, you should
    hold the state in a different way, and to react to the moves, you need to reset the board to the state you
    want to simulate.
    TODO: set_board, get_board, to_plane
    """
    def __init__(self, init=None):
        self.height = 10
        self.width = 9
        self.board = [['.' for col in range(self.width)] for row in range(self.height)]
        self.steps = 0
        self._legal_moves = None
        self.turn = RED
        self.winner = None
        if init is None or init == '':
            self.assign_fen(None)
        else:
            self.parse_init(init)

    def _update(self):
        self._fen = None
        self._legal_moves = None
        self.steps += 1
        if self.steps % 2 == 0:
            self.turn = RED
        else:
            self.turn = BLACK

    def parse_init(self, init):
        pieces = 'rnbakabnrccpppppRNBAKABNRCCPPPPP'
        position = [init[i:i+2] for i in range(len(init)) if i % 2 == 0]
        for pos, piece in zip(position, pieces):
            if pos != '99':
                x, y = int(pos[0]), 9 - int(pos[1])
                self.board[y][x] = piece

    def assign_board(self, board, turn=RED):
        self.board = board
        self.turn = turn

    def assign_fen(self, fen):
        if fen is None:
            fen = init_fen
        x = 0
        y = 0
        for k in range(0, len(fen)):
            ch = fen[k]
            if ch == ' ':
                if (fen[k+1] == 'b'):
                    self.turn = BLACK
                break
            if ch == '/':
                x = 0
                y += 1
            elif ch >= '1' and ch <= '9':
                for i in range(int(ch)):
                    self.board[y][x] = '.'
                    x = x + 1
            else:
                self.board[y][x] = ch
                x = x + 1

    def get_plane(self):
        return ChessBoard.s_get_plane(self.board, self.turn)

    def FENboard(self):
        return ChessBoard.sFENboard(self.board, self.turn)

    def fliped_FENboard(self):
        fen = self.FENboard()
        foo = fen.split(' ')
        rows = foo[0].split('/')
        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a
        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(reversed(row)) for row in reversed(rows)]) \
            + " " + foo[1] \
            + " " + foo[2] \
            + " " + foo[3] + " " + foo[4] + " " + foo[5]

    @property
    def is_red_turn(self):
        return self.turn == RED

    @property
    def screen(self):
        return self.board

    def legal_moves(self):
        if self._legal_moves is not None:
            return self._legal_moves

        _legal_moves = []
        for y in range(self.height):
            for x in range(self.width):
                ch = self.board[y][x]
                if (self.turn == RED and ch.isupper()):
                    continue
                if (self.turn == BLACK and ch.islower()):
                    continue
                if ch in mov_dir:
                    if (x == 0 and y == 3):
                        aa = 3
                    for d in mov_dir[ch]:
                        x_ = x + d[0]
                        y_ = y + d[1]
                        if not self._can_move(x_, y_):
                            continue
                        elif ch == 'p' and y < 5 and x_ != x:  # for red pawn
                            continue
                        elif ch == 'P' and y > 4 and x_ != x:  # for black pawn
                            continue
                        elif ch == 'n' or ch == 'N' or ch == 'b' or ch == 'B': # for knight and bishop
                            if self.board[y+int(d[1]/2)][x+int(d[0]/2)] != '.':
                                continue
                            elif ch == 'b' and y_ > 4:
                                continue
                            elif ch == 'B' and y_ < 5:
                                continue
                        elif ch != 'p' and ch != 'P': # for king and advisor
                            if x_ < 3 or x_ > 5:
                                continue
                            if (ch == 'k' or ch == 'a') and y_ > 2:
                                continue
                            if (ch == 'K' or ch == 'A') and y_ < 7:
                                continue
                        _legal_moves.append(move_to_str(x, y, x_, y_))
                        if (ch == 'k' and self.turn == RED): #for King to King check
                            d, u = self._y_board_from(x, y)
                            if (u < self.height and self.board[u][x] == 'K'):
                                _legal_moves.append(move_to_str(x, y, x, u))
                        elif (ch == 'K' and self.turn == BLACK):
                            d, u = self._y_board_from(x, y)
                            if (d > -1 and self.board[d][x] == 'k'):
                                _legal_moves.append(move_to_str(x, y, x, d))
                elif ch != '.': # for connon and root
                    l,r = self._x_board_from(x,y)
                    d,u = self._y_board_from(x,y)
                    for x_ in range(l+1,x):
                        _legal_moves.append(move_to_str(x, y, x_, y))
                    for x_ in range(x+1,r):
                        _legal_moves.append(move_to_str(x, y, x_, y))
                    for y_ in range(d+1,y):
                        _legal_moves.append(move_to_str(x, y, x, y_))
                    for y_ in range(y+1,u):
                        _legal_moves.append(move_to_str(x, y, x, y_))
                    if ch == 'r' or ch == 'R': # for root
                        if self._can_move(l, y):
                            _legal_moves.append(move_to_str(x, y, l, y))
                        if self._can_move(r, y):
                            _legal_moves.append(move_to_str(x, y, r, y))
                        if self._can_move(x, d):
                            _legal_moves.append(move_to_str(x, y, x, d))
                        if self._can_move(x, u):
                            _legal_moves.append(move_to_str(x, y, x, u))
                    else: # for connon
                        l_, _ = self._x_board_from(l,y)
                        _, r_ = self._x_board_from(r,y)
                        d_, _ = self._y_board_from(x,d)
                        _, u_ = self._y_board_from(x,u)
                        if self._can_move(l_, y):
                            _legal_moves.append(move_to_str(x, y, l_, y))
                        if self._can_move(r_, y):
                            _legal_moves.append(move_to_str(x, y, r_, y))
                        if self._can_move(x, d_):
                            _legal_moves.append(move_to_str(x, y, x, d_))
                        if self._can_move(x, u_):
                            _legal_moves.append(move_to_str(x, y, x, u_))

        self._legal_moves = _legal_moves
        return _legal_moves

    def is_legal(self, mov):
        return mov.uci in self.legal_moves

    def is_end(self):
        red_k, black_k = [0, 0], [0, 0]
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == 'k':
                    red_k[0] = i
                    red_k[1] = j
                if self.board[i][j] == 'K':
                    black_k[0] = i
                    black_k[1] = j
        if red_k[0] == 0 and red_k[1] == 0:
            self.winner = BLACK#Winner.black
        elif black_k[0] == 0 and black_k[1] == 0:
            self.winner = RED#Winner.red
        elif red_k[1] == black_k[1]:
            has_block = False
            i = red_k[0] + 1
            while i < black_k[0]:
                if self.board[i][red_k[1]] != '.':
                    has_block = True
                    break
                i += 1
            if not has_block:
                if self.turn == RED:
                    self.winner = RED
                else:
                    self.winner = BLACK
        return self.winner is not None

    def print_to_cl(self):
        for i in range(9, -1, -1):
            print(self.board[i])

    def move_action_str(self, uci):
        mov = Move(uci)
        self.push(mov)
        return True

    def move(self, mov):
        self.push(mov)

    def push(self, mov):
        self.board[mov.n[1]][mov.n[0]] = self.board[mov.p[1]][mov.p[0]]
        self.board[mov.p[1]][mov.p[0]] = '.'
        self._update()


    def _is_same_side(self,x,y):
        if self.turn == RED and self.board[y][x].islower():
            return True
        if self.turn == BLACK and self.board[y][x].isupper():
            return True

    def _can_move(self,x,y): # basically check the move
        if x < 0 or x > self.width-1:
            return False
        if y < 0 or y > self.height-1:
            return False
        if self._is_same_side(x,y):
            return False
        return True

    def _x_board_from(self,x,y):
        l = x-1
        r = x+1
        while l > -1 and self.board[y][l] == '.':
            l = l-1
        while r < self.width and self.board[y][r] == '.':
            r = r+1
        return l,r

    def _y_board_from(self,x,y):
        d = y-1
        u = y+1
        while d > -1 and self.board[d][x] == '.':
            d = d-1
        while u < self.height and self.board[u][x] == '.':
            u = u+1
        return d,u

    def result(self, claim_draw=True) -> str:
        rst = '*'
        if ('k' not in self.board[0]) and ('k' not in self.board[1]) and ('k' not in self.board[2]):
            rst = '0-1'
        if ('K' not in self.board[9]) and ('K' not in self.board[8]) and ('K' not in self.board[7]):
            rst = '1-0'
        return rst

    def clear_chessmans_moving_list(self):
        return

    def calc_chessmans_moving_list(self):
        return

    def save_record(self, filename):
        return

    def parse_WXF_move(self, wxf):
        '''
        red is upper, black is lower alphabet
        '''
        p = self.swapcase(wxf[0])
        col = wxf[1]
        mov = wxf[2]
        dest_col = wxf[3]
        src_row, src_col = self.find_row(p, col)
        if mov == '.' or mov == '=':
            # move horizontally
            dest_row = src_row
            if p.islower():
                dest_col = int(dest_col) - 1
            else:
                dest_col = self.width - int(dest_col)
        else:
            if p == 'h' or p == 'H' or p == 'e' or p == 'E' or p == 'a' or p == 'A':
                if p.islower():
                    dest_col = int(dest_col) - 1
                else:
                    dest_col = self.width - int(dest_col)

                if p == 'h' or p == 'H':
                    # for house/knight
                    step = 1 if abs(dest_col - src_col) == 2 else 2
                elif p == 'e' or p == 'E':
                    # for elephant/bishop
                    step = 2
                else:
                    # for advisor
                    step = 1 
                if mov == '+' and p.islower() or mov == '-' and p.isupper():
                    dest_row = src_row + step
                else:
                    dest_row = src_row - step
            else:
                # move vertically
                step = int(dest_col)
                if mov == '+' and p.islower() or mov == '-' and p.isupper():
                    dest_row = src_row + step
                else:
                    dest_row = src_row - step
                dest_col = src_col
        return move_to_str(src_col, src_row, dest_col, dest_row)

    def find_row(self, piece, col):
        if piece == 'h' or piece == 'H':
            piece = 'n' if piece == 'h' else 'N'
        if piece == 'e' or piece == 'E':
            piece = 'b' if piece == 'e' else 'B'
        column = 0
        row = -1
        if col.isdigit():
            if piece.isupper():
                column = self.width - int(col)
            else:
                column = int(col) - 1
            for i in range(self.height):
                if self.board[i][int(column)] == piece:
                    row = i
                    break
        else:
            first_row = -1
            second_row = -1
            column = -1
            for j in range(self.width):
                column = -1
                for i in range(self.height):
                    if self.board[i][j] == piece:
                        if column == -1:
                            column = j
                            first_row = i
                        else:
                            if column == j:
                                second_row = i
                                break
                            else:
                                column = j
                                first_row = second_row = -1
                if first_row != -1 and second_row != -1:
                    break
            if (piece.islower() and col == '+') or (piece.isupper() and col == '-'):
                row = second_row
            else:
                row = first_row
        return row, column

    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    
    def has_attack_chessman(self):
        for row in self.board:
            for chessman in row:
                c = chessman.lower()
                if c in ['r', 'n', 'p', 'c']:
                    return True
        return False
    
    """
    static part
    """
    def sFENboard(brd, turn):
        """
        Red side is on the bottom half, black side is on the top half, except when flipped
        """
        def swapcase(a):
            if a.isalpha():
                a = replace_dict[a]
                return a.lower() if a.isupper() else a.upper()
            return a

        c = 0
        fen = ''
        height = len(brd)
        width = len(brd[0])
        for i in range(height - 1, -1, -1):
            c = 0
            for j in range(width):
                if brd[i][j] == '.':
                    c = c + 1
                else:
                    if c > 0:
                        fen = fen + str(c)
                    fen = fen + swapcase(brd[i][j])
                    c = 0
            if c > 0:
                fen = fen + str(c)
            if i > 0:
                fen = fen + '/'
        if turn is RED:
            fen += ' r'
        else:
            fen += ' b'
        fen += ' - - 0 1'
        return fen

    def sfliped_FENboard(brd, turn):
        fen = ChessBoard.sFENboard(brd, turn)
        foo = fen.split(' ')
        rows = foo[0].split('/')
        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a
        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(reversed(row)) for row in reversed(rows)]) \
            + " " + foo[1] \
            + " " + foo[2] \
            + " " + foo[3] + " " + foo[4] + " " + foo[5]

    def s_get_plane(brd, turn):
        height = len(brd)
        width = len(brd[0])
        plane = np.zeros((14, height, width), dtype=np.float32)
        if turn == RED:
            for y in range(height):
                for x in range(width):
                    piece = brd[y][x]
                    if piece != '.':
                        plane[piece_to_plane[piece]][y][x] = 1
        else:
            # pretend to be red
            for y in range(height):
                for x in range(width):
                    piece = brd[height - y - 1][width - x - 1]
                    if piece != '.':
                        piece = ChessBoard.swapcase(piece)
                        plane[piece_to_plane[piece]][y][x] = 1
        return plane

    def create_action_labels():
        labels_array = []   # [col_src,row_src,col_dst,row_dst]
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # row
        letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8'] # col

        for n1 in range(10):
            for l1 in range(9):
                destinations = [(n1, t) for t in range(9)] + \
                            [(t, l1) for t in range(10)] + \
                            [(n1 + a, l1 + b) for (a, b) in
                                [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
                for (n2, l2) in destinations:
                    if (n1, l1) != (n2, l2) and n2 in range(10) and l2 in range(9):
                        move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                        labels_array.append(move)

        #for red mandarin
        labels_array.append('3041')
        labels_array.append('5041')
        labels_array.append('3241')
        labels_array.append('5241')
        labels_array.append('4130')
        labels_array.append('4150')
        labels_array.append('4132')
        labels_array.append('4152')
        # for black mandarin
        labels_array.append('3948')
        labels_array.append('5948')
        labels_array.append('3748')
        labels_array.append('5748')
        labels_array.append('4839')
        labels_array.append('4859')
        labels_array.append('4837')
        labels_array.append('4857')

        #for red elephant
        labels_array.append('2002')
        labels_array.append('2042')
        labels_array.append('6042')
        labels_array.append('6082')
        labels_array.append('2402')
        labels_array.append('2442')
        labels_array.append('6442')
        labels_array.append('6482')
        labels_array.append('0220')
        labels_array.append('4220')
        labels_array.append('4260')
        labels_array.append('8260')
        labels_array.append('0224')
        labels_array.append('4224')
        labels_array.append('4264')
        labels_array.append('8264')
        # for black elephant
        labels_array.append('2907')
        labels_array.append('2947')
        labels_array.append('6947')
        labels_array.append('6987')
        labels_array.append('2507')
        labels_array.append('2547')
        labels_array.append('6547')
        labels_array.append('6587')
        labels_array.append('0729')
        labels_array.append('4729')
        labels_array.append('4769')
        labels_array.append('8769')
        labels_array.append('0725')
        labels_array.append('4725')
        labels_array.append('4765')
        labels_array.append('8765')

        return labels_array
    
    def flip_move(x):
        new = ''.join([str(8 - int(x[0])),
                       str(9 - int(x[1])),
                       str(8 - int(x[2])),
                       str(9 - int(x[3]))
                      ])
        return new
    
    def orig_move(x, player):
        if player == RED:
            return x
        else:
            return ChessBoard.flip_move(x)

    def print_board_impl(board, move_from=None, move_to=None, indent="    ", level=logging.INFO):
        """
        Print the Chinese Chess board in ASCII with colors.
        :param board: The current board state.
        :param move_from: The position (row, col) of the piece being moved.
        :param move_to: The destination position (row, col) of the move.
        """
        # Define the board layout with borders and labels
        logger.log(level, indent + "    a    b    c    d    e    f    g    h    i")
        logger.log(level, indent + " +----+----+----+----+----+----+----+----+----+")
        
        for i in range(10):
            i_ = 9 - i
            row = str(i_) + "|"
            for j in range(9):
                piece = board[i_][j]
                piece1 = PIECES.get(piece, piece)
                # Highlight the "move_from" and "move_to" positions
                if (i_, j) == move_from:
                    row += f" {Back.RED}{Fore.WHITE}{piece1}{Style.RESET_ALL} |"
                elif (i_, j) == move_to:
                    row += f" {Back.GREEN}{Fore.WHITE}{piece1}{Style.RESET_ALL} |"
                else:
                    # Color Red pieces in red and Black pieces in blue
                    if piece.isupper():  # Red pieces
                        row += f" {Fore.GREEN}{piece1}{Style.RESET_ALL} |"
                    elif piece.islower():  # Black pieces
                        row += f" {Fore.RED}{piece1}{Style.RESET_ALL} |"
                    else:  # Empty spaces
                        row += f" {piece1} |"
            logger.log(level, indent + row)
            logger.log(level, indent + " +----+----+----+----+----+----+----+----+----+")
        
        logger.log(level, indent + "    a    b    c    d    e    f    g    h    i")

    def initialize_board():
        """
        Initialize the Chinese Chess board with starting positions.
        """
        # Create a 10x9 grid (rows x columns)
        board = [['·' for _ in range(9)] for _ in range(10)]
        
        # Red pieces
        board[0] = ['R', 'H', 'E', 'A', 'K', 'A', 'E', 'H', 'R']
        board[2] = ['·', 'C', '·', '·', '·', '·', '·', 'C', '·']
        board[3] = ['S', '·', 'S', '·', 'S', '·', 'S', '·', 'S']
        
        # Black pieces
        board[9] = ['r', 'h', 'e', 'a', 'k', 'a', 'e', 'h', 'r']
        board[7] = ['·', 'c', '·', '·', '·', '·', '·', 'c', '·']
        board[6] = ['s', '·', 's', '·', 's', '·', 's', '·', 's']
        
        return board

    def infer_move(board1, board2):
        """
        Infer the move (move_from and move_to) by comparing board1 and board2.
        :param board1: The board before the move.
        :param board2: The board after the move.
        :return: A tuple (move_from, move_to) representing the move.
        """
        move_from = None
        move_to = None
        for i in range(10):
            for j in range(9):
                if board1[i][j] != board2[i][j]:
                    print(board1[i][j], board2[i][j])
                    if board1[i][j] != '·':  # Piece moved from here
                        move_from = (i, j)
                    if board2[i][j] != '·':  # Piece moved to here
                        move_to = (i, j)
        
        return move_from, move_to
    
    def print_board(prev_board, cur_board, indent="    ", level=logging.INFO):
        move_from, move_to = None, None
        if prev_board is not None:
            move_from, move_to = ChessBoard.infer_move(prev_board, cur_board)
        ChessBoard.print_board_impl(cur_board, move_from, move_to, indent=indent, level=level)

    def flip_board_and_players(board):
        """
        Flip the Chinese Chess board and the players.
        """
        # Flip the board and swap the players
        # Create new board with flipped positions
        flipped_board = [['.' for _ in range(9)] for _ in range(10)]
        for i in range(10):
            for j in range(9):
                # Get piece at original position
                piece = board[i][j]
                if piece.isalpha():
                    # Flip piece color (upper/lower case)
                    piece = piece.lower() if piece.isupper() else piece.upper()
                # Place piece in flipped position 
                flipped_board[9-i][8-j] = piece
        return flipped_board

    def parse_visualization(visualization):
        """
        Convert a visualization of the Chinese Chess board back into its internal representation.
        :param visualization: A list of strings representing the board visualization.
        :return: A 10x9 grid representing the internal board state.
        """
        board = [['.' for _ in range(9)] for _ in range(10)]
        visualization = visualization.split('\n')
        # Skip the first two lines (headers and borders)
        for i in range(2, 22, 2):  # Rows are printed every 2 lines
            row = visualization[i].strip().split('|')[1:]  # Split by '|' and ignore the first element
            for j in range(9):
                piece = row[j].strip()  # Remove extra spaces
                if piece in REVERSE_PIECES:
                    board[(i // 2) - 1][j] = REVERSE_PIECES[piece]
        board = list(reversed(board))
        return board

    def read_visualization_from_file(file_path):
        """
        Read the visualization of the Chinese Chess board from a file and parse it.
        :param file_path: Path to the file containing the board visualization.
        :return: A 10x9 grid representing the internal board state.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            visualization = file.read()
        return ChessBoard.parse_visualization(visualization)
    
    def hash_board(raw_board):
        # NOTE: there can be confliction but it's good for debugging
        # return abs(hash(repr(raw_board))) % 10000
        #return abs(hash(tuple(map(tuple, raw_board)))) % 10000
        s = ''.join([''.join(r) for r in raw_board])
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)%10000
    
    def adjudicate_by_pieces_for_red(board):
        piece_vals = {'N': 3, 'K': 14, 'R': 5, 'C': 3.25, 'E': 2, 'A':2, 'P': 1}
        ans = 0.0
        tot = 0
        for r in board:
            for c in r:
                if not c.isalpha():
                    continue

                if not c.isupper():
                    ans += piece_vals[c.upper()]
                    tot += piece_vals[c.upper()]
                else:
                    ans -= piece_vals[c]
                    tot += piece_vals[c]
        v = ans/tot
        #assert abs(v) < 1
        return np.tanh(v * 3) # arbitrary

label_actions = ChessBoard.create_action_labels()
action_labels = {move: i for move, i in zip(label_actions, range(len(label_actions)))}



if __name__ == '__main__': # test
    board = ChessBoard()
    board.turn = BLACK
    board.move_action_str('0304')
    board.print_to_cl()
    print(board.FENboard())
    print(board.fliped_FENboard())
    print(board.legal_moves())