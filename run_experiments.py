# -*- coding: utf-8 -*-
"""
Пункт 2 задания: исследование работы ГА на трёх тестовых функциях
нелинейного программирования (Sphere, Rastrigin, Rosenbrock), включая оценку
эффективности. Каждая функция исследуется в двух размерностях (n=2 — чтобы
сопоставить с визуализацией, и n=10 — чтобы показать масштабируемость на
"настоящую" задачу от n аргументов), по 5 независимых запусков на конфигурацию.
"""
import json
from ga_nd import (
    SphereFunction, RastriginFunction, RosenbrockFunction,
    BinaryEncoder, GeneticAlgorithm, GAConfig,
    RouletteSelection, SinglePointCrossover, BitFlipMutation,
)
from ga_nd.experiment import ExperimentRunner

SEEDS = [1, 2, 3, 4, 5]
FUNCTIONS = [SphereFunction, RastriginFunction, RosenbrockFunction]
DIMENSIONS = [2, 10]

rows = []
details = {}

for cls in FUNCTIONS:
    for n in DIMENSIONS:
        obj = cls(n=n)

        def make_ga(obj=obj):
            encoder = BinaryEncoder(obj.bounds, precision=1e-3)
            config = GAConfig(
                population_size=max(60, 20 * obj.n),   # больше особей для большей размерности
                pc=0.85,
                max_generations=400,
                stagnation_limit=60,
                mode="min",
            )
            return GeneticAlgorithm(obj, encoder, RouletteSelection(),
                                     SinglePointCrossover(), BitFlipMutation(), config)

        runner = ExperimentRunner(make_ga, obj, eps_success=1e-2)
        summary = runner.run(SEEDS)
        row = summary.to_row()
        rows.append(row)
        details[f"{obj.name}_n{n}"] = [
            {"seed": r.seed, "f_found": r.f_found, "error": r.error,
             "success": r.success, "generations": r.generations,
             "evaluations": r.evaluations}
            for r in summary.records
        ]
        print(f"{row['function']:10s} n={row['n']:2d}  "
              f"success={row['success_rate_%']:5.1f}%  "
              f"mean_error={row['mean_error']:.5f}  "
              f"mean_gen={row['mean_generations']:6.1f}  "
              f"mean_eval={row['mean_evaluations']:7.1f}")

with open("/mnt/user-data/outputs/experiment_results.json", "w", encoding="utf-8") as f:
    json.dump({"summary": rows, "details": details}, f, ensure_ascii=False, indent=2)

print("\nСохранено: experiment_results.json")
