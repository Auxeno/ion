"""Shared analysis helpers for benchmark reports and plots."""

import statistics
from collections import defaultdict
from dataclasses import replace

from .protocol import Result


def balance_results(results: list[Result]) -> list[Result]:
    """Match repetitions and sample counts across compared series."""
    comparisons = defaultdict(lambda: defaultdict(list))
    for result in results:
        comparison = result.model, result.size, result.metric
        series = result.framework, result.mode
        comparisons[comparison][series].append(result)

    balanced = []
    for series_results in comparisons.values():
        repetitions = min(map(len, series_results.values()))
        selected = [
            result
            for values in series_results.values()
            for result in values[:repetitions]
        ]
        sample_counts = [len(result.samples_ms) for result in selected if result.samples_ms]
        samples_per_repetition = min(sample_counts, default=0)

        for result in selected:
            if not samples_per_repetition:
                balanced.append(result)
                continue
            samples = result.samples_ms[:samples_per_repetition]
            throughput = result.throughput
            if throughput is not None:
                throughput = result.units_per_step / (statistics.median(samples) / 1000)
            balanced.append(replace(result, samples_ms=samples, throughput=throughput))
    return balanced
