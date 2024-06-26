from typing import Dict

import pint

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
        self.model_units: Dict[str, str] = {}
        self.parameter_units: Dict[str, str] = {}
        self.observable_units: Dict[str, str] = {}
        self.measurement_units: Dict[str, str] = {}
        self.condition_units: Dict[str, str] = {}

    def petab_units(self):
        """Resolve all units for a given problem."""

        # parameter
        df = self.problem.parameter_df
        units = {}
        if PARAMETER_UNIT_COLUMN in df.columns:
            for oid, row in df.iterrows():
                units[oid] = row[PARAMETER_UNIT_COLUMN]
        self.parameter_units = units
        console.print(f"{self.parameter_units=}")

        # observable
        df = self.problem.observable_df
        units = {}
        observable2formula = {}
        if OBSERVABLE_UNIT_COLUMN in df.columns:
            for oid, row in df.iterrows():
                formula = row["observableFormula"]
                observable2formula[oid] = formula
                units[formula] = row[OBSERVABLE_UNIT_COLUMN]

        console.print(f"{observable2formula=}")
        self.observable_units = units
        console.print(f"{self.observable_units=}")

        # measurement
        df = self.problem.measurement_df
        units = {}
        if MEASUREMENT_UNIT_COLUMN in df.columns:
            console.print(df.observableId.unique())
            for observable_id in df.observableId.unique():
                formula = observable2formula[oid]
                print(observable_id, formula)
                df_observable = df[df.observableId == observable_id]
                units[formula] = list(df_observable[MEASUREMENT_UNIT_COLUMN].unique())

        if MEASUREMENT_TIME_UNIT_COLUMN in df.columns:
            units["time"] = list(df_observable[MEASUREMENT_TIME_UNIT_COLUMN].unique())

        self.measurement_units = units
        console.print(f"{self.measurement_units=}")

        # check the condition dataframe
        console.rule("conditions", style="white")
        df = self.problem.parameter_df
        units = {}
        if PARAMETER_UNIT_COLUMN in df.columns:
            for oid, row in df.iterrows():
                units[oid] = row[PARAMETER_UNIT_COLUMN]
        self.parameter_units = units
        console.rule("parameters", style="white")
        console.print(df)
        console.print(f"{self.parameter_units=}")


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
        petab_uparser.petab_units()



