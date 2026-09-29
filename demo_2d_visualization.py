# -*- coding: utf-8 -*-
"""
Наглядная визуализация работы ГА для функции от ДВУХ аргументов (п.1 задания):
для каждой из трёх тестовых функций строится
  (a) сетка снимков популяции на фоне линий уровня функции по поколениям,
  (b) график сходимости (лучшее / среднее значение по поколениям).
"""
import os
from ga_nd import (
    SphereFunction, RastriginFunction, RosenbrockFunction,
    BinaryEncoder, GeneticAlgorithm, GAConfig,
    RouletteSelection, SinglePointCrossover, BitFlipMutation,
)
from ga_nd.visualizer import GAVisualizer2D

OUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

FUNCTIONS = [SphereFunction, RastriginFunction, RosenbrockFunction]

for cls in FUNCTIONS:
    obj = cls(n=2)
    encoder = BinaryEncoder(obj.bounds, precision=1e-3)
    config = GAConfig(population_size=60, pc=0.85, max_generations=150,
                       stagnation_limit=40, mode="min")
    ga = GeneticAlgorithm(obj, encoder, RouletteSelection(),
                           SinglePointCrossover(), BitFlipMutation(), config)

    result = ga.run(seed=7, track_population=True)
    print(f"{obj.name}: x*={result.best_x}, f*={result.best_f:.6f}, "
          f"generations={result.generations}, evaluations={result.evaluations}")

    viz = GAVisualizer2D(obj, grid_points=200)
    snap_path = viz.plot_population_snapshots(
        result.history_population,
        filename=f"{OUT_DIR}/{obj.name.lower()}_population_evolution.png")
    conv_path = viz.plot_convergence(
        result, filename=f"{OUT_DIR}/{obj.name.lower()}_convergence.png")
    print(f"  сохранено: {snap_path}, {conv_path}")
