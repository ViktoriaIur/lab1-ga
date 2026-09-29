
"""
operators.py
------------
Генетические операторы реализованы как отдельные классы-стратегии
(паттерн Strategy). Это даёт:

  * SRP — каждый класс отвечает за один оператор;
  * OCP — новый вид кроссинговера/мутации/селекции добавляется новым классом,
    без изменения GeneticAlgorithm;
  * ISP — три узких интерфейса (Selection, Crossover, Mutation) вместо одного
    "толстого" интерфейса оператора;
  * DIP — GeneticAlgorithm получает конкретные стратегии через конструктор
    (внедрение зависимостей), а не создаёт их сам и не завязан на их тип.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


# =============================================================================
# СЕЛЕКЦИЯ
# =============================================================================

class Selection(ABC):
    @abstractmethod
    def select(self, chromosomes: List[np.ndarray], scores: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        """Возвращает ОДНУ выбранную хромосому. `scores` — веса для
        максимизации (чем больше, тем лучше особь; неотрицательные)."""
        raise NotImplementedError


class RouletteSelection(Selection):
    """Пропорциональная селекция (рулетка): вероятность выбора особи
    пропорциональна её весу."""

    def select(self, chromosomes, scores, rng):
        total = scores.sum()
        if total <= 0:
            idx = rng.integers(0, len(chromosomes))
        else:
            probs = scores / total
            idx = rng.choice(len(chromosomes), p=probs)
        return chromosomes[idx]


class TournamentSelection(Selection):
    """Турнирная селекция: из k случайных особей побеждает лучшая.
    Менее чувствительна к «суперлидерам» на ранних поколениях, чем рулетка."""

    def __init__(self, k: int = 3):
        self.k = k

    def select(self, chromosomes, scores, rng):
        idxs = rng.integers(0, len(chromosomes), size=self.k)
        best_idx = idxs[np.argmax(scores[idxs])]
        return chromosomes[best_idx]


# =============================================================================
# КРОССИНГОВЕР
# =============================================================================

class Crossover(ABC):
    @abstractmethod
    def apply(self, parent1: np.ndarray, parent2: np.ndarray,
              pc: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SinglePointCrossover(Crossover):
    def apply(self, parent1, parent2, pc, rng):
        if rng.random() < pc and len(parent1) > 1:
            point = rng.integers(1, len(parent1))
            c1 = np.concatenate([parent1[:point], parent2[point:]])
            c2 = np.concatenate([parent2[:point], parent1[point:]])
            return c1, c2
        return parent1.copy(), parent2.copy()


class UniformCrossover(Crossover):
    """Каждый бит потомка независимо берётся от одного из родителей
    с вероятностью 0.5 — сильнее перемешивает гены, чем одноточечный."""

    def apply(self, parent1, parent2, pc, rng):
        if rng.random() < pc:
            mask = rng.integers(0, 2, size=len(parent1)).astype(bool)
            c1 = np.where(mask, parent1, parent2)
            c2 = np.where(mask, parent2, parent1)
            return c1, c2
        return parent1.copy(), parent2.copy()


# =============================================================================
# МУТАЦИЯ
# =============================================================================

class Mutation(ABC):
    @abstractmethod
    def apply(self, chromosome: np.ndarray, pm: float,
              rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class BitFlipMutation(Mutation):
    """Каждый бит независимо инвертируется с вероятностью pm."""

    def apply(self, chromosome, pm, rng):
        mask = rng.random(len(chromosome)) < pm
        result = chromosome.copy()
        result[mask] = 1 - result[mask]
        return result
