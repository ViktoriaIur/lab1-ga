
"""
encoding.py
-----------
Отвечает ТОЛЬКО за преобразование "вектор вещественных чисел <-> двоичная
хромосома" (SRP). GeneticAlgorithm ничего не знает о битах — он работает
только через интерфейс Encoder (DIP: зависимость от абстракции, а не от
конкретной схемы кодирования; можно подставить, например, Грея-код или
вещественное кодирование, унаследовав от Encoder).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class Encoder(ABC):
    @abstractmethod
    def chromosome_length(self) -> int: ...

    @abstractmethod
    def random_chromosome(self, rng: np.random.Generator) -> np.ndarray: ...

    @abstractmethod
    def decode(self, chromosome: np.ndarray) -> np.ndarray: ...


class BinaryEncoder(Encoder):
    """Классическое позиционное двоичное кодирование.

    Хромосома — конкатенация n сегментов бит (по одному на координату).
    Длина сегмента i вычисляется из требуемой точности `precision` и границ
    `bounds[i]`, так чтобы шаг дискретизации не превышал precision:
        (hi - lo) / (2**bits - 1) <= precision
    """

    def __init__(self, bounds: List[Tuple[float, float]], precision: float = 1e-3):
        self.bounds = bounds
        self.n = len(bounds)
        self.precision = precision
        self._bits_per_var = [self._bits_for(lo, hi, precision) for lo, hi in bounds]
        offsets = [0]
        for b in self._bits_per_var:
            offsets.append(offsets[-1] + b)
        self._offsets = offsets  # длина n+1

    @staticmethod
    def _bits_for(lo: float, hi: float, precision: float) -> int:
        need = (hi - lo) / precision
        bits = 1
        while (2 ** bits - 1) < need:
            bits += 1
        return bits

    def chromosome_length(self) -> int:
        return self._offsets[-1]

    def random_chromosome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=self.chromosome_length(), dtype=np.uint8)

    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        x = np.empty(self.n, dtype=float)
        for i, (lo, hi) in enumerate(self.bounds):
            start, end = self._offsets[i], self._offsets[i + 1]
            segment = chromosome[start:end]
            as_int = 0
            for bit in segment:
                as_int = (as_int << 1) | int(bit)
            max_int = 2 ** self._bits_per_var[i] - 1
            x[i] = lo + as_int * (hi - lo) / max_int
        return x
