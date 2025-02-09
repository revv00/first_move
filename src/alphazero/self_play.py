from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from config import config
from mcts import MCTS
from env.chessboard import ChessBoard
from env.common import RED, BLACK
from config import config

class EndType(Enum):
    WIN_LOSE = 1
    LOOP = 2
    NONE_ATTACKER = 3

def adjudicate(history, board):
    if board.is_end():
        if board.turn == RED:
            return True, EndType.WIN_LOSE, 1.0
        else:
            return True, EndType.WIN_LOSE, -1.0
    else:
        # if no attacker
        if not board.has_attack_chessman():
            return True, EndType.NONE_ATTACKER, 0.0
        # if loop
        free_move = defaultdict(int)
        for brd, turn, policy in history[:-1]:
            free_move[brd] += 1
            if free_move[brd] >= 3:
                return True, EndType.LOOP, 0.0
        return False, None, 0.0

def play_a_game(config):
    # Chess environment with current side to move
    board = ChessBoard()
    mcts = MCTS(config.self_play)
    # Take turns to play moves until the game ends, each move itself is a MCTS simulation
    while board.winner is None:
        # Get the current player
        player = board.turn

        # Perform MCTS simulation
        action = mcts.mcts_srch(board, player)
        if player == BLACK:
            action = ChessBoard.flip_move(action)
        # Perform the action on the board
        board.move_action_str(action)
        value = 0
        if board.steps/2 > config.max_game_length:
            value = ChessBoard.adjudicate_by_pieces(board)
            break
        else:
            is_end, end_type, value = adjudicate(mcts._history, board)
            if is_end:
                break


if __name__ == '__main__':
    futures = deque()
    parallel = config.self_play.game_num
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        futures.append(executor.submit(play_a_game, config))
        for future in futures:
            result = future.result()
        