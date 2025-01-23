import sys
import copy
import queue
import math
import random
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from threading import Lock
from env.common import *
from env import chessboard
from env.chessboard import ChessBoard, action_labels, Winner
from model.client import ModelClient

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        #self._model_clients = ThreadPoolExecutor(max_workers=self._config.num_clients)
        self._client_queue = queue.Queue()
        # TODO: uncomment this
        #for _ in range(self._config.num_clients):
        #    self._client_queue.put(ModelClient())

    def mcts_srch(self, board, player, can_stop=True):
        # reset for every actual move
        self._tree = defaultdict(StateStats)
        with ThreadPoolExecutor(max_workers=self._config.mcts_sims) as executor:
            futures = [executor.submit(self.mcts_srch_once, copy.deepcopy(board), player) for _ in range(self._config.mcts_sims)]
            vals = [future.result() for future in futures]
        policy = self.solve_policy(board)
        my_action = int(np.random.choice(range(self.labels_n), p = self.apply_temperature(policy, board.steps/2)))
        root_value = max(vals)
        if can_stop and self._config.resign_threshold is not None and \
                        root_value <= self._config.resign_threshold \
                        and board.steps/2 > self._config.min_resign_turn:
            # noinspection PyTypeChecker
            return None
        else:
            self._history.append([copy.deepcopy(board.board), board.turn, list(policy)])
            return chessboard.label_actions[my_action]


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
        logger.debug("Action selecting...")
        for i, (move, stat) in enumerate(ss._as.items()):
            print(state, chessboard.label_actions[move], stat)
            p = stat.p if not is_root else (1 - e) * stat.p + e * noise[i]
            v = stat.q + w_p * p * sqrt_num / (1 + stat.n)
            if v > best_v:
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
            logger.debug(f"Find end state {board.board}, {v}")
            return None, v
        plane = board.get_plane()
        dirichlet_distribution = np.random.dirichlet([1.0] * len(action_labels))
        return dirichlet_distribution, random.choice([-0.1, 0, 0.1])
        # As if it is red
        """
        model_client = self._client_queue.get()
        try:
            p, v = model_client.evaluate()
        finally:
            self._client_queue.put(model_client)
        return p, v
        """

    def mcts_srch_once(self, board, player, level=0):
        """
        Evaluate as if it's red, Store statistics as if it's red, But move as what it really is.
        """
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
                        for m in board.legal_moves():
                            m = ChessBoard.flip_move(m) if board.turn == BLACK else m
                            p_all += ps[action_labels[m]]
                            legal_mvs.append(action_labels[m])
                        for m in legal_mvs:
                            stats._as[m].p = ps[m] / p_all
                    stats.v = v
                    stats.ps = ps
                logger.debug(f"Find unk state {board.board}, {stats._as}, {v}")
                return stats.v
            # Explore and Exploit
            best_move_idx, _ = self.action_selection_as_is_red(state, level==0)
            if best_move_idx == -1:
                # no longer up propagate
                return self._tree[state].v
            
            # hack to discount q so that other threads will explore more
            stats = self._tree[state]
            if level > self._config.max_depth:
                stats.v = 0.0
                stats.ps = None
            else: 
                as_ = stats._as[best_move_idx]
                stats.sum_n += vl
                as_.n += vl
                as_.w -= vl
                as_.q = as_.w/as_.n
        
        v = 0
        if not (stats.v == 0.0 and stats.ps is None):
            mv = ChessBoard.orig_move(chessboard.label_actions[best_move_idx], player)
            logger.debug(f"Action select for {player} at srch depth {level}: {mv}, {self._tree[state]._as}")
            # NOTE: for debug TODO: remove 
            bef_board = copy.deepcopy(board.board)
            board.move_action_str(mv)
            logger.debug(f"Move after action select: {mv}")
            ChessBoard.print_board(None, board.board)
            v = -self.mcts_srch_once(board, board.turn, level+1)

            # Up propagate the stats
            with self._locks[state]:
                stats.sum_n += -vl + 1
                as_.n += -vl + 1
                as_.w += vl + v
                as_.q = as_.w/as_.n
            logger.debug(f"player:{player}, depth:{level}, board_before_mv:{ChessBoard.hash_board(bef_board)}, mv:{mv} n:{as_.n} W:{as_.w}, Q:{as_.q}")
        return v

    def solve_policy(self, board):
        state = board.FENboard(board.turn) if board.turn is RED else board.fliped_FENboard()
        ss = self._tree[state]
        policy = np.zeros(len(action_labels))
        for move, as_ in ss._as.items():
            policy[action_labels[move]] = as_.n
        policy /= np.sum(policy)
        return policy
    
    def apply_temperature(self, policy, turn):
        tau = np.power(self._config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret