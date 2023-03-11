from math import log, exp
from time import time

from problem.queens_move import QueensMoveState
from problem.queens_swap import QueensSwapState
from algorithm.hill_climb import HillClimb
from utils.selection import MaxSelection, FirstBetterSelection, RouletteSelection

if __name__ == "__main__":
    n = 40
    log_n = log(n)
    queens_move_state_value_estimator = lambda state: exp(-log_n * state.conflicts)  # ==1/(n^conflicts)
    queens_swap_state_value_estimator = lambda state: exp(-log_n * state.state.conflicts)

    t0 = time()
    f_selection = FirstBetterSelection()  # 第一最佳选择 能找到的话 想必也是最快的
    r_selection = RouletteSelection()  # 轮盘赌选择
    m_selection = MaxSelection()  # 最大估值选择

    state = QueensMoveState(n)
    hcs = HillClimb(state)
    hcs.search(queens_move_state_value_estimator, 1.0, n << 2, f_selection, 5)

    # state = QueensSwapState(n)
    # hcs = HillClimb(state)
    # hcs.search(queens_swap_state_value_estimator, 1.0, n << 2, f_selection, 5)  # target_value是1.0 意味着conflict得为0

    print(time() - t0)
