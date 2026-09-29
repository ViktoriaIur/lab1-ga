
"""
visualizer.py
-------------
GAVisualizer2D — единственная задача (SRP): наглядно показать работу ГА
для n=2 (линии уровня функции + положение популяции по поколениям, и
график сходимости). GeneticAlgorithm ничего не знает о визуализации
(разделение ответственности), визуализатор ничего не знает о деталях
селекции/кроссинговера/мутации — работает только с результатом (GAResult)
и объектом функции (ObjectiveFunction).
"""

from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .objective_functions import ObjectiveFunction
from .genetic_algorithm import GAResult


class GAVisualizer2D:
    def __init__(self, objective: ObjectiveFunction, grid_points: int = 250):
        if objective.n != 2:
            raise ValueError("GAVisualizer2D работает только для функций от 2 аргументов")
        self.objective = objective
        self.grid_points = grid_points
        (self.x_lo, self.x_hi), (self.y_lo, self.y_hi) = objective.bounds
        self._build_contour_grid()

    def _build_contour_grid(self) -> None:
        xs = np.linspace(self.x_lo, self.x_hi, self.grid_points)
        ys = np.linspace(self.y_lo, self.y_hi, self.grid_points)
        X, Y = np.meshgrid(xs, ys)
        Z = np.empty_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.objective.evaluate(np.array([X[i, j], Y[i, j]]))
        self.X, self.Y, self.Z = X, Y, Z

    def _draw_contour(self, ax) -> None:
        ax.contourf(self.X, self.Y, self.Z, levels=40, cmap="viridis")
        x_star, _ = self.objective.global_optimum()
        ax.scatter([x_star[0]], [x_star[1]], color="white", marker="*",
                   s=150, edgecolor="black", zorder=5, label="Глобальный минимум")
        ax.set_xlim(self.x_lo, self.x_hi)
        ax.set_ylim(self.y_lo, self.y_hi)

    def plot_population_snapshots(self, history_population: List[np.ndarray],
                                   generations_to_show: Optional[List[int]] = None,
                                   filename: str = "population_snapshots.png") -> str:
        """Сетка снимков популяции на фоне линий уровня функции — наглядно
        показывает, как популяция стягивается к оптимуму по поколениям."""
        total = len(history_population)
        if generations_to_show is None:
            step = max(1, total // 6)
            generations_to_show = list(range(0, total, step))[:6]
            if generations_to_show[-1] != total - 1:
                generations_to_show[-1] = total - 1

        n_plots = len(generations_to_show)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, gen in zip(axes, generations_to_show):
            self._draw_contour(ax)
            pop = history_population[gen]
            ax.scatter(pop[:, 0], pop[:, 1], color="red", s=14,
                       alpha=0.8, zorder=4, label="Популяция")
            ax.set_title(f"Поколение {gen}")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        for ax in axes[n_plots:]:
            ax.axis("off")

        axes[0].legend(loc="upper right", fontsize=7)
        fig.suptitle(f"Эволюция популяции: функция {self.objective.name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(filename, dpi=140)
        plt.close(fig)
        return filename

    def plot_convergence(self, result: GAResult, filename: str = "convergence.png") -> str:
        """Кривые сходимости лучшего и среднего значений функции по поколениям."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(result.history_best, label="Лучшее значение f", color="crimson")
        ax.plot(result.history_avg, label="Среднее значение f по популяции",
                color="steelblue", alpha=0.7)
        ax.set_xlabel("Поколение")
        ax.set_ylabel("f(x)")
        ax.set_yscale("symlog")
        ax.set_title(f"Сходимость ГА: {self.objective.name}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=140)
        plt.close(fig)
        return filename
