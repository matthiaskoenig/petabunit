"""Running all examples and benchmark problems."""
from pathlib import Path
from typing import List

import pandas as pd

from petabunit import EXAMPLES_DIR
from petabunit.console import console
from petabunit.sbmlunits import SBMLUnitParser


def sbml_unit_statistics() -> pd.DataFrame:
    """Create unit statistics for all examples and benchmark problems."""
    sbml_paths_examples: List[Path] = sorted(
        [p for p in EXAMPLES_DIR.glob("**/*.xml")])
    BENCHMARK_DIR = Path(
        "/home/mkoenig/git/Benchmark-Models-PEtab/Benchmark-Models")
    sbml_paths_benchmarks: List[Path] = sorted(
        [p for p in BENCHMARK_DIR.glob("**/*.xml")]
    )
    sbml_paths = [p for p in sbml_paths_examples + sbml_paths_benchmarks]

    df = SBMLUnitParser.unit_statistics(sbml_paths=sbml_paths)
    df.to_csv(EXAMPLES_DIR / "statistics.tsv", sep="\t", index=False)
    console.rule(style="white")
    console.print(df)
    console.rule(style="white")
    return df


if __name__ == "__main__":
    sbml_unit_statistics()

