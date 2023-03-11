from abc import ABCMeta, abstractmethod
import sys

from .random_variables import RandomVariables


class SelectionBase(metaclass=ABCMeta):

    @abstractmethod
    def initialize(self, items: int, initial_value: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def submit(self, value: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def selected_index(self) -> int:
        raise NotImplementedError


class RouletteSelection(SelectionBase):

    def __init__(self):
        pass

    def initialize(self, items: int, initial_value: float) -> None:
        self._value_sum = 0
        self._index = 0
        self._total_items = items
        self._submitted_items = 0

    def submit(self, value: float) -> None:  # 需要遍历全部，每次选择会有权重 value/value_sum 与value成正比
        # 这种方式第k个被选择的概率是value_k/sum_value_k*sum_value_k/sum_value_k+1*...*sum_value_n-1/sum_value_n正好等于value_k在全部n个状态value和中所占比。
        assert value >= 0
        self._value_sum += value
        if RandomVariables.uniform_real() < value / (self._value_sum + sys.float_info.min):
            self._index = self._submitted_items
        self._submitted_items += 1

    def done(self) -> bool:
        return self._submitted_items >= self._total_items

    def selected_index(self) -> int:
        return self._index


class FirstBetterSelection(SelectionBase):

    def __init__(self):
        pass

    def initialize(self, items: int, initial_value: float) -> None:
        self._target_value = initial_value
        self._index = 0
        self._total_items = items
        self._submitted_items = 0
        self._item_found = False

    def submit(self, value: float) -> None:  # 只要出现大于原有value的状态，就item_found，记录index
        if value > self._target_value:
            self._item_found = True
            self._index = self._submitted_items

        self._submitted_items += 1

    def done(self) -> bool:
        return self._item_found or self._submitted_items >= self._total_items

    def selected_index(self) -> int:
        return self._index


class MaxSelection(SelectionBase):

    def __init__(self):
        pass

    def initialize(self, items: int, initial_value: float) -> None:
        self._index = 0
        self._total_items = items
        self._submitted_items = 0

    def submit(self, value: float) -> None:  # 需要全部遍历，记录最大value状态
        if self._submitted_items == 0 or value > self._max_value:
            self._max_value = value
            self._index = self._submitted_items

        self._submitted_items += 1

    def done(self) -> bool:
        return self._submitted_items >= self._total_items

    def selected_index(self) -> int:
        return self._index
