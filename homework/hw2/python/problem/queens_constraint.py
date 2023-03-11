from typing import List

from interface.constraint_satisfaction import ConstraintSatisfactionBase
from utils.random_variables import RandomVariables
from .queens_move import QueensMoveState


class QueensConstraintSatisfaction(QueensMoveState, ConstraintSatisfactionBase):

    def __init__(self, n: int):
        super().__init__(n)  # 这里面随机初始化了一个queens 是n的全排列
        self._choices = list(range(n))

    def n_variables(self) -> int:
        return self.n_queens

    def variables(self) -> List[int]:  # 存放了queens list的整数（对应每一行的列数）
        return self.queens

    def conflicts_of(self, variable_index: int) -> int:  # 计算某一行的queen冲突个数
        row, column = variable_index, self.queens[variable_index]
        return (self.column_count[column]
                + self.right_left_count[column + row]
                + self.left_right_count[column - row + self.n_queens - 1]
                - 3
                )

    def has_conflict(self) -> bool:
        return any(self.conflicts_of(i) for i in range(self.n_queens))
        # any函数 只要可迭代容器中有一个为True就返回True

    def choices_of(self, variable_index: int) -> List[int]:
        return self._choices

    def set_variable(self, variable_index: int, new_variable: float) -> None:  # 把index行的皇后放到new_variable列
        self._add_queen(variable_index, self.queens[variable_index], -1)
        self._add_queen(variable_index, new_variable)

    def show(self) -> None:
        QueensMoveState.show(self)

    def reset(self) -> None:
        QueensMoveState.reset(self)
