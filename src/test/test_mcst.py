import sys
import unittest
import copy
import numpy as np
from env.chessboard import ChessBoard, action_labels, label_actions
from env.common import RED, BLACK
from alphazero.mcts import MCTS
from config import config
from concurrent.futures import ThreadPoolExecutor

class TestChessBoard(unittest.TestCase):

    def setUp(self):
        self.board = ChessBoard()

    def print_stats(self, mcst, depths):
        state_distribution = {}
        for s, stats in mcst._tree.items():
            state_distribution[s] = stats.sum_n
        
        print(f'''State distribution in MCTS tree: {len(state_distribution)},
no_zero: {len({k:v for k,v in state_distribution.items() if v > 0})}
ge_10: {len({k:v for k,v in state_distribution.items() if v >= 10})}
Average depth: {sum(depths)/len(depths)}
''')
        for state, count in sorted(state_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"Count: {count}")

    @unittest.skipIf(True, "skip this test")
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
        depths = []
        for i in range(2000):
            d, _ = mcst.mcts_srch_once(copy.deepcopy(cb), RED)
            depths.append(d)
        self.print_stats(mcst, depths)

    @unittest.skipIf(True, "skip this test")
    def test_read_and_mcst_multithread(self):
        data_root = 'data/vboards/soldier_king1.txt'
        print("Test read board from file")
        board = ChessBoard.read_visualization_from_file(data_root)
        cb = ChessBoard()
        cb.assign_board(board, turn=RED)
        
        mcst = MCTS(config.self_play)
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(mcst.mcts_srch_once, copy.deepcopy(cb), RED) for _ in range(2000)]
            depths = [future.result()[0] for future in futures]
        self.print_stats(mcst, depths)

    @unittest.skipIf(True, "skip this test")
    def test_read_and_mcst_multithread1(self):
        data_root = 'data/vboards/check_or_not.txt'
        print("Test read board from file")
        board = ChessBoard.read_visualization_from_file(data_root)
        cb = ChessBoard()
        cb.assign_board(board, turn=BLACK)#RED
        
        mcst = MCTS(config.self_play)
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(mcst.mcts_srch_once, copy.deepcopy(cb)) for _ in range(2000)]
            depths = [future.result()[0] for future in futures]
        self.print_stats(mcst, depths)

    @unittest.skipIf(False, "skip this test")
    def test_read_and_mcst_multithread2(self):
        # test it: for i in `seq 1 40`; do python test/test_mcst.py 2>&1 | grep -P "Action:|Wrong"; done
        data_root = 'data/vboards/red_choice_strange.txt'
        print("Test read board from file")
        board = ChessBoard.read_visualization_from_file(data_root)
        cb = ChessBoard()
        cb.assign_board(board, turn=RED)#RED
        cb.steps = 18
        mcst = MCTS(config.self_play)
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(mcst.mcts_srch_once, copy.deepcopy(cb)) for _ in range(2000)]
            depths = [future.result()[0] for future in futures]
        self.print_stats(mcst, depths)
        policy = mcst.solve_policy(cb)
        ps = mcst.apply_temperature(policy, cb.steps/2)
        my_action = int(np.random.choice(range(len(action_labels)), p = ps))
        print(f"Action: {label_actions[my_action]} P: {policy[my_action]} P after temperature: {ps[my_action]}")
        print(f"Policy For Wrong Move: {policy[action_labels['2241']]} P after temperature: {ps[action_labels['2241']]}")
        #print(mcst.mcts_srch(cb))

if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    unittest.main()