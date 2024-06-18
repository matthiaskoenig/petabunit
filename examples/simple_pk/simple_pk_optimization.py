import petab
from pathlib import Path
import numpy as np
import pypesto.petab
from petabunit.console import console
from matplotlib import pyplot as plt
import logging

console.rule("Load PEtab", style="white")
petab_yaml: Path = Path(__file__).parent / "simple_pk.yaml"
petab_problem = petab.Problem.from_yaml(petab_yaml)
importer = pypesto.petab.PetabImporter(petab_problem)
problem = importer.create_problem(verbose=True)

console.rule("observables", style="white")
console.print(petab_problem.observable_df)


