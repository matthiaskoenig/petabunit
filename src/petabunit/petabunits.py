import petab
from petabunit import EXAMPLES_DIR
from petabunit.console import console
from petabunit import log

logger = log.get_logger(__name__)

PARAMETER_UNIT_COLUMN = "parameterUnit"
OBSERVABLE_UNIT_COLUMN = "observableUnit"
MEASUREMENT_UNIT_COLUMN = "measurementUnit"
MEASUREMENT_TIME_UNIT_COLUMN = "timeUnit"
CONDITION_UNIT_COLUMN_SUFFIX = "Unit"


class PEtabUnitParser:
    """Parser for PEtab unit information."""

    def __init__(self, problem: petab.Problem) -> None:
        self.problem = problem

    @staticmethod
    def units_for_petab_problem(problem: petab.Problem):
        """Resolve all units for a given problem."""

        console.rule("parameters", style="white")
        parameter_df = problem.parameter_df

        # Check the observable dataframe
        console.rule("observables", style="white")
        console.print(problem.observable_df)

        # Check the measurement dataframe
        console.rule("measurements", style="white")
        console.print(problem.measurement_df)

        # check the condition dataframe
        console.rule("conditions", style="white")
        console.print(problem.condition_df)


    @staticmethod
    def print_problem(problem: petab.Problem) -> None:
        console.rule("parameters", style="white")
        console.print(problem.parameter_df)

        # Check the observable dataframe
        console.rule("observables", style="white")
        console.print(problem.observable_df)

        # Check the measurement dataframe
        console.rule("measurements", style="white")
        console.print(problem.measurement_df)

        # check the condition dataframe
        console.rule("conditions", style="white")
        console.print(problem.condition_df)

        # change things in the model
        console.rule(style="white")


if __name__ == "__main__":
    yaml_paths = [p for p in EXAMPLES_DIR.glob("**/*.yaml")]
    yaml_paths = [
        p for p in (EXAMPLES_DIR / "simple_chain").glob("*.yaml")
    ]
    for yaml_path in yaml_paths:
        console.rule(yaml_path.stem)
        problem: petab.Problem = petab.Problem.from_yaml(yaml_path)
        console.print(problem)
        errors_exist = petab.lint.lint_problem(problem)
        console.print(f"PEtab errors: {errors_exist}")

        petab_uparser = PEtabUnitParser(problem)
        PEtabUnitParser.print_problem(problem)



