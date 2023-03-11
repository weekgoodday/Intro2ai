from math import exp
from time import time

from problem.queens_constraint import QueensConstraintSatisfaction
from algorithm.conflicts_minimize import ConflictMinimize
from utils.selection import MaxSelection, FirstBetterSelection, RouletteSelection

if __name__ == '__main__':
    t0 = time()
    alpha = 10
    value_of = lambda value: exp(alpha * value)  # 以exp(10输入)作为估值 输入都是某行冲突数

    n = 1600
    q = QueensConstraintSatisfaction(n)
    cm = ConflictMinimize(q)

    fbs = FirstBetterSelection()
    rs = RouletteSelection()
    ms = MaxSelection()

    cm.search(10, n << 2, ms)

    print(f"Total time: {time() - t0}")
