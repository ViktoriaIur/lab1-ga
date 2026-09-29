
"""
population.py
--------------
Population инкапсулирует коллекцию особей и предоставляет над ней операции
(лучшая особь, средняя приспособленность и т.д.). GeneticAlgorithm работает
с популяцией через эти методы, а не лезет во внутренний список напрямую —
это упрощает будущую замену структуры хранения (например, на numpy-матрицу)
без изменения кода GeneticAlgorithm (DRY + инкапсуляция).
"""

from typing import List
import numpy as np

from .individual import Individual


class Population:
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def fitness_array(self) -> np.ndarray:
        return np.array([ind.fitness for ind in self.individuals])

    def raw_value_array(self) -> np.ndarray:
        return np.array([ind.raw_value for ind in self.individuals])

    def best(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.fitness)

    def average_fitness(self) -> float:
        return float(np.mean(self.fitness_array()))

    def positions(self) -> np.ndarray:
        """Матрица (pop_size x n) декодированных координат — используется
        визуализатором для отображения популяции на графике."""
        return np.array([ind.x for ind in self.individuals])
