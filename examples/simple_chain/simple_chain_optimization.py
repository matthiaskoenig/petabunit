"""
pip install amici
pip install pypesto
pip install pypesto[fides]
"""
import petab
import pypesto
from pathlib import Path
import numpy as np
import pypesto.petab
from petabunit.console import console

console.rule("Load PEtab", style="white")
petab_yaml: Path = Path(__file__).parent / "simple_chain.yaml"
petab_problem = petab.Problem.from_yaml(petab_yaml)
importer = pypesto.petab.PetabImporter(petab_problem)
problem = importer.create_problem(verbose=False)

# Check the observable dataframe
console.print(petab_problem.observable_df.head())

# Check the measurement dataframe
console.print(petab_problem.measurement_df.head())

# check the condition dataframe
console.print(petab_problem.condition_df.head())

# change things in the model
console.print(problem.objective.amici_model.requireSensitivitiesForAllParameters())
# change solver settings
console.print(
    f"Absolute tolerance before change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
)
problem.objective.amici_solver.setAbsoluteTolerance(1e-15)
console.print(
    f"Absolute tolerance after change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
)

optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}
import logging
optimizer = pypesto.optimize.FidesOptimizer(
    options=optimizer_options, verbose=logging.WARN
)
startpoint_method = pypesto.startpoint.uniform
# save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)

opt_options = pypesto.optimize.OptimizeOptions()
console.print(opt_options)

n_starts = 20  # usually a value >= 100 should be used
engine = pypesto.engine.MultiProcessEngine()

# Set seed for reproducibility
np.random.seed(1)

console.rule("Optimization", style="white")
result = pypesto.optimize.minimize(
    problem=problem,
    optimizer=optimizer,
    n_starts=n_starts,
    startpoint_method=startpoint_method,
    engine=engine,
    options=opt_options,
)
console.print(result.summary())

ax = pypesto.model_fit.visualize_optimized_model_fit(
    petab_problem=petab_problem, result=result, pypesto_problem=problem
)

pypesto.visualize.waterfall(result);
pypesto.visualize.parameters(result);




if __name__ == "__main__":
    pass
