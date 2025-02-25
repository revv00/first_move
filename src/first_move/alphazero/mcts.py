import os
import copy
import queue
import math
import random
import logging
import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from threading import Lock


from ..config import PolicyType
from ..env.common import *
from ..env import chessboard
from ..env.chessboard import ChessBoard, action_labels
from ..model.client import ModelClient
# from model.client import ModelClient

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(thread)d - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(os.getenv('CHESS_LOG_LEVEL', logging.ERROR))

class StateStats:
    def __init__(self):
        self._as = defaultdict(ActionStats)
        self.sum_n = 0
        self.v = None
        self.ps = None
        #self.p = None

class ActionStats:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

class MCTS:
    """
    Include both players' thoughts
    """
    def __init__(self, config):
        self._tree = defaultdict(StateStats)
        self._locks = defaultdict(Lock)
        self._config = config
        self._history = []
        self._print_lock = Lock()
        #self._model_clients = ThreadPoolExecutor(max_workers=self._config.num_clients)
        self._client_queue = queue.Queue()
        self._current_policy = PolicyType.RANDOM
        self._current_policy_iter = 0
        for _ in range(self._config.num_clients):
            self._client_queue.put(ModelClient())

    def __del__(self):
        while not self._client_queue.empty():
            client = self._client_queue.get()
            client.close()

    def mcts_srch(self, board, can_stop=True):
        # reset for every actual move
        self._tree = defaultdict(StateStats)
        player = board.turn
        self._current_policy = self._config.red_type if player == RED else self._config.black_type
        self._current_policy_iter = self._config.red_iter if player == RED else self._config.black_iter
        with ThreadPoolExecutor(max_workers=self._config.rollout_parallelism) as executor:
            futures = [executor.submit(self.mcts_srch_once, copy.deepcopy(board)) for _ in range(self._config.mcts_sims)]
            vals = [future.result()[1] for future in futures]
        policy = self.solve_policy(board)
        my_action = int(np.random.choice(range(len(action_labels)), p = self.apply_temperature(policy, board.steps/2)))
        root_value = max(vals)
        if can_stop and self._config.resign_threshold is not None and \
                        root_value <= self._config.resign_threshold \
                        and board.steps/2 > self._config.min_resign_turn:
            # noinspection PyTypeChecker
            return None, None, None
        else:
            # NOTE: policy is for as if it is red, so we need to flip the board when training
            self._history.append([copy.deepcopy(board.board), board.turn, list(policy)])
            return chessboard.label_actions[my_action] if player == RED \
                else ChessBoard.flip_move(chessboard.label_actions[my_action]), policy[my_action], board.steps/2


    def action_selection_as_is_red(self, state, is_root=False):
        ss = self._tree[state]
        e = self._config.noise_eps
        dir_alpha = self._config.dir_alpha
        w_p = self._config.wgt_p
        if is_root:
            noise = np.random.dirichlet([dir_alpha] * len(ss._as))
        best_move = -1
        best_v = -1000
        sqrt_num = math.sqrt(ss.sum_n + 1)
        with self._print_lock:
            logger.debug("Action selecting...")
        for i, (move, stat) in enumerate(ss._as.items()):
            p = stat.p if not is_root else (1 - e) * stat.p + e * noise[i]
            v = stat.q + w_p * p * sqrt_num / (1 + stat.n)
            # NOTE: if the good move is in the front of the list, it will be very probable to be replaced 
            if v == best_v:
                if random.random() < 0.5:
                    best_v = v
                    best_move = move
            elif v > best_v:
                best_v = v
                best_move = move
        return best_move, best_v 

    def evaluate(self, board, level):
        # NOTE: return random for testing
        # TODO: remove this
        if level > self._config.max_depth:
            return None, 0.0
        if board.is_end():
            v = self._config.win_reward if (board.turn == board.winner) else \
                (-self._config.win_reward if (board.winner is not None) else 0.0)
            with self._print_lock:
                logger.debug(f"Find end state {board.board}, {v}")
            return None, v
        #dirichlet_distribution = np.random.dirichlet([1.0] * len(action_labels))
        if self._current_policy == PolicyType.RANDOM:
            distribution = np.ones(len(action_labels)) / len(action_labels) \
                if self._config.is_random_policy_uniform == 1 else \
                np.random.multinomial(
                    len(action_labels),
                    np.ones(len(action_labels))/len(action_labels)
                ) / len(action_labels)
            return distribution, random.choice([-0.1, 0, 0.1])
        else:
            plane = board.get_plane()
            model_client = self._client_queue.get()
            try:
                p, v = model_client.predict(self._current_policy_iter, plane)
            except:
                traceback.print_exc()
                # degrade to randomm policy
                logger.error("Degrade to a random policy")
                p = np.ones(len(action_labels)) / len(action_labels) 
                v = random.choice([-0.1, 0, 0.1])
            finally:
                self._client_queue.put(model_client)
            return p, v

    def mcts_srch_once(self, board, level=0):
        """
        Evaluate as if it's red, Store statistics as if it's red, But move as what it really is.
        """
        player = board.turn
        state = board.FENboard() if player is RED else board.fliped_FENboard()
        vl = self._config.virtual_loss
        with self._locks[state]:
            # Expland and Eval if leaf node
            if state not in self._tree:
                stats = self._tree[state]
                # if it is end
                if stats.v is None:
                    ps, v = self.evaluate(board, level)
                    p_all = 0
                    legal_mvs = []
                    if ps is not None:
                        with self._print_lock:
                            logger.info(f"{board.turn} {level} {ChessBoard.hash_fen_code(state)} {len(board.legal_moves())}")
                        for m in random.sample(board.legal_moves(), len(board.legal_moves())):
                            m = ChessBoard.flip_move(m) if board.turn == BLACK else m
                            p_all += ps[action_labels[m]]
                            legal_mvs.append(action_labels[m])
                        for m in legal_mvs:
                            stats._as[m].p = ps[m] / (p_all + 1e-9)
                    stats.v = v
                    stats.ps = ps
                with self._print_lock:
                    logger.debug(f"Find unk state {ChessBoard.hash_board(board.board)}, {stats._as}, {v}")
                return level, stats.v
            # Explore and Exploit
            best_move_label, _ = self.action_selection_as_is_red(state, level==0)
            if best_move_label == -1:
                # no longer up propagate
                return level, self._tree[state].v
            
            # hack to discount q so that other threads will explore more
            stats = self._tree[state]
            if level > self._config.max_depth:
                stats.v = 0.0
                stats.ps = None
            else: 
                as_ = stats._as[best_move_label]
                stats.sum_n += vl
                as_.n += vl
                as_.w -= vl
                as_.q = as_.w/as_.n
        
        v = 0
        depth = level
        if not (stats.v == 0.0 and stats.ps is None):
            mv = ChessBoard.orig_move(chessboard.label_actions[best_move_label], player)
            with self._print_lock:
                logger.debug(f"L{level} Action select for {player}: {mv}")
            # NOTE: for debug TODO: remove 
            bef_board = copy.deepcopy(board.board)
            board.move_action_str(mv)
            with self._print_lock:
                logger.debug(f"L{level} Move after action select: {mv}")
                ChessBoard.print_board(None, board.board)
            depth, v = self.mcts_srch_once(board, level+1)
            v = -v

            # Up propagate the stats
            with self._locks[state]:
                stats.sum_n += -vl + 1
                as_.n += -vl + 1
                as_.w += vl + v
                as_.q = as_.w/as_.n
            with self._print_lock:
                logger.debug(f"L{level} player:{player},  board_before_mv:{ChessBoard.hash_board(bef_board)}, mv:{mv} n:{as_.n} W:{as_.w}, Q:{as_.q}")
        else:
            logger.debug("searching too deep")
        return depth, v

    def solve_policy(self, board):
        state = board.FENboard() if board.turn is RED else board.fliped_FENboard()
        ss = self._tree[state]
        logger.info(f"state: {state} (hash: {ChessBoard.hash_fen_code(state)}) for {board.turn} and visit_num:{ss.sum_n}")
        policy = np.zeros(len(action_labels))
        for move_label, as_ in ss._as.items():
            logger.info(f"as_red_mov: {chessboard.label_actions[move_label]} {as_.n} {as_.q}")
            policy[move_label] = as_.n
        policy /= np.sum(policy)
        return policy
    
    def apply_temperature(self, policy, turn):
        tau = np.power(self._config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(len(policy))
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def update_history_with_red_value(self, red_value):
        for i in range(len(self._history)):
            self._history[i] += [red_value if self._history[i][1] == RED else -red_value]
    
    def save_history(self, iteration, task_id):
        df = pd.DataFrame(self._history, columns=['board', 'turn', 'policy', 'value'])
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists(f'./data/train/iter_{iteration}'):
            os.makedirs(f'./data/train/iter_{iteration}')
        filename = f'./data/train/iter_{iteration}/{task_id}_{timestamp}.parquet'
        df.to_parquet(filename, engine='pyarrow')
    
    def clear_history(self):
        self._history = []