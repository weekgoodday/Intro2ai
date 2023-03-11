from math import exp, log
from time import time

from problem.queens_move import QueensMoveState
from problem.queens_swap import QueensSwapState
from algorithm.simulated_anneal import SimulatedAnneal

if __name__ == '__main__':
    n = 67
    log_n = log(n)
    start_temp_log = n / log(n)
    max_conflicts = n * (n - 1) >> 1
    value_estimator_swap = lambda state: max_conflicts - state.state.conflicts
    value_estimator_move = lambda state: max_conflicts - state.conflicts

    temperature_schedule_move = lambda time: exp(start_temp_log - time / (n << 4))
    temperature_schedule_swap = lambda time: exp(start_temp_log - time / n)

    t0 = time()

    # q = QueensSwapState(n)
    # sa = SimulatedAnneal(q)
    # sa.search(value_estimator_swap, temperature_schedule_swap, n << 2, max_conflicts, 1e-16)

    q = QueensMoveState(n)
    sa = SimulatedAnneal(q)
    sa.search(value_estimator_move, temperature_schedule_move, n << 2, max_conflicts, 1e-16)
    # terminate_temperature为常数1e-16 基本要达到16n*(n+16)才到停止探索，可忽略

    print(f"Total time: {time() - t0}")
