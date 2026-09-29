
"""
objective_functions.py
-----------------------
Абстракция целевой (тестовой) функции нелинейного программирования.

Принцип SOLID, реализуемый здесь:
  * OCP (Open/Closed) — чтобы добавить новую тестовую функцию, достаточно
    унаследоваться от ObjectiveFunction и не нужно менять ни один другой
    класс библиотеки (GeneticAlgorithm, Visualizer и т.д. работают только
    через этот интерфейс).
  * SRP (Single Responsibility) — класс отвечает ТОЛЬКО за вычисление
    значения функции и знание своих границ/эталона, никакой логики ГА тут нет.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class ObjectiveFunction(ABC):
    """Базовый интерфейс тестовой функции f: R^n -> R."""

    name: str = "ObjectiveFunction"

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("Размерность n должна быть >= 1")
        self.n = n

    @property
    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]:
        """Границы поиска по каждой координате: [(min_1,max_1), ..., (min_n,max_n)]."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Значение функции в точке x (вектор длины n)."""
        raise NotImplementedError

    @abstractmethod
    def global_optimum(self) -> Tuple[np.ndarray, float]:
        """Известное (эталонное) решение — используется только для оценки
        эффективности ГА в вычислительных экспериментах, самим ГА не используется."""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)

    def __repr__(self) -> str:
        return f"{self.name}(n={self.n})"


# =============================================================================
# Три тестовые функции нелинейного программирования разной "сложности":
#   1) Sphere      — унимодальная, выпуклая (эталон "лёгкой" задачи)
#   2) Rosenbrock  — узкий изогнутый овраг (проверка способности к точной
#                    доводке решения, классический сложный тест для ГА)
#   3) Rastrigin   — сильно мультимодальная (проверка способности избегать
#                    застревания в локальных экстремумах)
# =============================================================================

class SphereFunction(ObjectiveFunction):
    """f(x) = sum(x_i^2). Глобальный минимум f=0 в x=(0,...,0)."""
    name = "Sphere"

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(-5.12, 5.12)] * self.n

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def global_optimum(self) -> Tuple[np.ndarray, float]:
        return np.zeros(self.n), 0.0


class RastriginFunction(ObjectiveFunction):
    """f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)). Глобальный минимум f=0 в x=0.
    Сотни локальных минимумов вокруг глобального — сложный тест на exploration."""
    name = "Rastrigin"
    A = 10.0

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(-5.12, 5.12)] * self.n

    def evaluate(self, x: np.ndarray) -> float:
        return float(self.A * self.n + np.sum(x ** 2 - self.A * np.cos(2 * np.pi * x)))

    def global_optimum(self) -> Tuple[np.ndarray, float]:
        return np.zeros(self.n), 0.0


class RosenbrockFunction(ObjectiveFunction):
    """f(x) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2). Глобальный минимум f=0 в x=(1,...,1).
    Узкая параболическая долина — легко найти долину, сложно точно сойтись к минимуму."""
    name = "Rosenbrock"

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(-2.048, 2.048)] * self.n

    def evaluate(self, x: np.ndarray) -> float:
        if self.n < 2:
            raise ValueError("Rosenbrock определена для n >= 2")
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def global_optimum(self) -> Tuple[np.ndarray, float]:
        return np.ones(self.n), 0.0
