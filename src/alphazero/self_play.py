import os, sys
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from config import config
from .mcts import MCTS
from env.chessboard import ChessBoard
from env.common import RED, BLACK
from config import config

class EndType(Enum):
    WIN_LOSE = 1
    LOOP = 2
    NONE_ATTACKER = 3

def adjudicate_for_red(history, board):
    if board.is_end():
        if board.winner == RED:
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
            k = ChessBoard.sFENboard(brd, turn) if turn == RED else ChessBoard.sfliped_FENboard(brd, turn)
            free_move[k] += 1
            if free_move[k] >= 3:
                return True, EndType.LOOP, 0.0
        return False, None, 0.0

def play_a_game(config, iteration=0, task_id=0):
    # Chess environment with current side to move
    board = ChessBoard()
    mcts = MCTS(config.self_play)
    # Take turns to play moves until the game ends, each move itself is a MCTS simulation
    while board.winner is None:
        # Get the current player
        player = board.turn

        # Perform MCTS simulation
        action = mcts.mcts_srch(board, player)

        # Perform the action on the board
        board.move_action_str(action)
        print(f"Turns: {board.steps}, Player: {player}, Action: {action}")
        ChessBoard.print_board(None, board.board, indent='')
        value = 0
        if board.steps/2 > config.self_play.max_game_length:
            value = ChessBoard.adjudicate_by_pieces_for_red(board)
            break
        else:
            is_end, _, value = adjudicate_for_red(mcts._history, board)
            if is_end:
                break
    # Update history for training, NOTE: the board should be then flipped for red because the policy is for red
    mcts.update_history_with_red_value(value)
    mcts.save_history(iteration, task_id)


if __name__ == '__main__':
    futures = deque()
    iteration = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    tasks = config.self_play.game_num
    parallel = os.cpu_count()
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        for i in range(tasks):
            futures.append(executor.submit(play_a_game, config, iteration, i))
        for future in futures:
            result = future.result()
        