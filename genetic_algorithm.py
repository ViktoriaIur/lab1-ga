
"""
genetic_algorithm.py
---------------------
GeneticAlgorithm — единственный класс, который "знает" про цикл поколений.
Он НЕ знает деталей кодирования, селекции, кроссинговера и мутации — все
эти детали приходят в конструктор как объекты-стратегии (Dependency
Inversion Principle). Благодаря этому класс:
  * не меняется при добавлении новой тестовой функции (работает через
    ObjectiveFunction);
  * не меняется при добавлении нового оператора (работает через Selection /
    Crossover / Mutation);
  * не меняется при смене схемы кодирования (работает через Encoder).

Это и есть практическая реализация SOLID для данной задачи.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .objective_functions import ObjectiveFunction
from .encoding import Encoder
from .operators import Selection, Crossover, Mutation
from .individual import Individual
from .population import Population


@dataclass
class GAConfig:
    """Гиперпараметры ГА — вынесены в отдельный dataclass (SRP: конфигурация
    отделена от логики выполнения)."""
    population_size: int = 60
    pc: float = 0.85                 # вероятность кроссинговера
    pm: Optional[float] = None       # вероятность мутации бита (None -> 1/L)
    max_generations: int = 300
    stagnation_limit: int = 40       # условие останова "застой"
    target_eps: float = 1e-5
    elitism: bool = True
    mode: str = "min"                # "min" или "max"


@dataclass
class GAResult:
    best_x: np.ndarray
    best_f: float
    generations: int
    evaluations: int
    history_best: List[float] = field(default_factory=list)
    history_avg: List[float] = field(default_factory=list)
    history_population: Optional[List[np.ndarray]] = None  # для визуализации


class GeneticAlgorithm:
    def __init__(self,
                 objective: ObjectiveFunction,
                 encoder: Encoder,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 config: GAConfig):
        self.objective = objective
        self.encoder = encoder
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.config = config

    # -------------------------------------------------------------------
    def _score(self, raw_value: float) -> float:
        """Перевод значения целевой функции в приспособленность (для
        максимизации операторами селекции). При minimize инвертируем знак."""
        return -raw_value if self.config.mode == "min" else raw_value

    def _is_better(self, a: float, b: float) -> bool:
        """True, если значение a лучше значения b с учётом режима min/max."""
        return a < b if self.config.mode == "min" else a > b

    def _evaluate_individual(self, ind: Individual) -> None:
        ind.x = self.encoder.decode(ind.chromosome)
        ind.raw_value = self.objective.evaluate(ind.x)
        ind.fitness = self._score(ind.raw_value)

    def _init_population(self, rng: np.random.Generator) -> Population:
        individuals = [
            Individual(chromosome=self.encoder.random_chromosome(rng))
            for _ in range(self.config.population_size)
        ]
        for ind in individuals:
            self._evaluate_individual(ind)
        return Population(individuals)

    # -------------------------------------------------------------------
    def run(self, seed: Optional[int] = None,
            track_population: bool = False) -> GAResult:
        cfg = self.config
        rng = np.random.default_rng(seed)
        pm = cfg.pm if cfg.pm is not None else 1.0 / self.encoder.chromosome_length()

        population = self._init_population(rng)
        evaluations = len(population)

        best = population.best().copy()
        history_best: List[float] = []
        history_avg: List[float] = []
        history_population: Optional[List[np.ndarray]] = [] if track_population else None
        stagnation = 0
        generation = 0

        for generation in range(1, cfg.max_generations + 1):
            gen_best = population.best()
            improved = gen_best.fitness > best.fitness + cfg.target_eps
            if gen_best.fitness > best.fitness:
                best = gen_best.copy()

            history_best.append(best.raw_value)
            history_avg.append(float(np.mean(population.raw_value_array())))
            if track_population:
                history_population.append(population.positions())

            stagnation = 0 if improved else stagnation + 1
            if stagnation >= cfg.stagnation_limit:
                break

            population = self._next_generation(population, rng, pm)
            evaluations += len(population)

        return GAResult(
            best_x=best.x, best_f=best.raw_value,
            generations=generation, evaluations=evaluations,
            history_best=history_best, history_avg=history_avg,
            history_population=history_population,
        )

    # -------------------------------------------------------------------
    def _next_generation(self, population: Population,
                          rng: np.random.Generator, pm: float) -> Population:
        cfg = self.config
        chromosomes = [ind.chromosome for ind in population]
        scores = population.fitness_array()
        # приведение весов селекции к неотрицательным (нужно рулетке)
        shifted = scores - scores.min() + 1e-9

        new_individuals: List[Individual] = []
        if cfg.elitism:
            new_individuals.append(population.best().copy())

        while len(new_individuals) < cfg.population_size:
            p1 = self.selection.select(chromosomes, shifted, rng)
            p2 = self.selection.select(chromosomes, shifted, rng)
            c1, c2 = self.crossover.apply(p1, p2, cfg.pc, rng)
            c1 = self.mutation.apply(c1, pm, rng)
            c2 = self.mutation.apply(c2, pm, rng)
            for c in (c1, c2):
                if len(new_individuals) < cfg.population_size:
                    ind = Individual(chromosome=c)
                    self._evaluate_individual(ind)
                    new_individuals.append(ind)

        return Population(new_individuals)
