
"""
individual.py
-------------
Individual — простой контейнер данных одной особи (KISS: никакой логики
кроме хранения; декодирование и вычисление фитнеса выполняют Encoder и
ObjectiveFunction — Individual о них ничего не знает, что соответствует SRP).
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Individual:
    chromosome: np.ndarray
    x: Optional[np.ndarray] = None          # декодированный вектор решения
    raw_value: Optional[float] = None        # значение f(x) "как есть"
    fitness: Optional[float] = None          # приспособленность (для максимизации)

    def copy(self) -> "Individual":
        return Individual(
            chromosome=self.chromosome.copy(),
            x=None if self.x is None else self.x.copy(),
            raw_value=self.raw_value,
            fitness=self.fitness,
        )
