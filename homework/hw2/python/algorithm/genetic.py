from typing import List

from interface.population import PopulationBase


class GeneticAlgorithm:
    ChromosomeType = PopulationBase.ChromosomeType

    def __init__(self, init_population: PopulationBase):
        self.population = init_population

    def return_population(self) -> List[ChromosomeType]:  # 尽量不要与成员变量同名
        return self.population.population()

    def adaptability(self) -> List[float]:
        return self.population.adaptability()

    def evolve(self, n: int) -> None:
        for i in range(n):
            print(f"<epoch {i} begin>")
            self.population.cross()
            if self.population.end_signal:
                self.population.show()
                for index, j in enumerate(self.adaptability()):
                    if j == 0:
                        print(f"Search end with index: {index}, Queens: {self.return_population()[index]}")
                        break
                break
            self.population.mutate()
            if self.population.end_signal:
                self.population.show()
                for index, j in enumerate(self.adaptability()):
                    if j == 0:
                        print(f"Search end with index: {index}, Queens: {self.return_population()[index]}")
                        break
                break
            self.population.show()
            print(f"<epoch {i} end>")

