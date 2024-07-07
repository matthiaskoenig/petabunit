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

# check tbe observables df
console.rule("observables", style="white")
console.print(petab_problem.observable_df)

# Check the measurement dataframe
console.rule("measurements", style="white")
console.print(petab_problem.measurement_df)

# check the condition dataframe
console.rule("conditions", style="white")
console.print(petab_problem.condition_df)

# change things in the model
console.rule(style="white")
console.print(problem.objective.amici_model.requireSensitivitiesForAllParameters())

print(
    f"Absolute tolerance before change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
)
problem.objective.amici_solver.setAbsoluteTolerance(1e-15)
print(
    f"Absolute tolerance after change: {problem.objective.amici_solver.getAbsoluteTolerance()}"
)

optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}

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
console.rule("results", style="white")
console.print(result.summary())
fig_path: Path = Path(__file__).parent / "results"


import pypesto.visualize.model_fit as model_fit
ax = model_fit.visualize_optimized_model_fit(
    petab_problem=petab_problem, result=result, pypesto_problem=problem
)
pypesto.visualize.waterfall(result)
plt.savefig(str(fig_path) + '/01_waterfall.png')

pypesto.visualize.parameters(result)
plt.savefig(str(fig_path) + '/02_parameters.png')

# pypesto.visualize.parameters_correlation_matrix(result)
console.rule("Parameter_hist", style="white")
# pypesto.visualize.parameter_hist(result=result, parameter_name="kabs")
# plt.savefig(str(fig_path) + '/03_parameters_hist.png')

pypesto.visualize.optimization_scatter(result)
plt.savefig(str(fig_path) + '/04_opt_scatter.png')

console.rule("Profile", style="white")
result = pypesto.profile.parameter_profile(
    problem=problem,
    result=result,
    optimizer=optimizer,
    engine=engine,
    # profile_index=[0, 1],
)

pypesto.visualize.profiles(result)
plt.savefig(str(fig_path) + '/05_profiles.png')

console.rule("Sampler", style="white")
sampler = pypesto.sample.AdaptiveMetropolisSampler()
result = pypesto.sample.sample(
    problem=problem,
    sampler=sampler,
    n_samples=5000,
    result=result,
)

pypesto.visualize.sampling_fval_traces(result)
plt.tight_layout()
plt.savefig(str(fig_path) + '/06_sampling_fval_traces.png')

pypesto.visualize.sampling_1d_marginals(result)
plt.savefig(str(fig_path) + '/07_marginals.png')
plt.show()




