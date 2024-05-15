"""Example of unit checking and conversion."""
import libsbml

from petabunit.console import console
from petabunit.units import unit_statistics_for_doc, unit_statistics

examples = [
    "Elowitz_Nature2000/Elowitz_Nature2000.yaml",  # no units
    "simple_chain/simple_chain.yaml",  # complete example with units
]


def example_Elowitz2000() -> None:
    """Run Elowitz2000 example."""
    console.rule("Elowitz2000", style="white")


import petab
def print_problem(petab_problem: petab.Problem) -> None:

    console.rule("parameters", style="white")
    console.print(petab_problem.parameter_df)

    # Check the observable dataframe
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

if __name__ == "__main__":
    from petabunit import EXAMPLES_DIR

    sbml_paths = [p for p in EXAMPLES_DIR.glob("**/*.xml")]
    console.print(f"{sbml_paths}")
    yaml_paths = [p for p in EXAMPLES_DIR.glob("**/*.yaml")]

    yaml_paths = [
        EXAMPLES_DIR / "baymodts_caffeine" / "caffeine_pk_Control.yaml"
    ]
    # unit statistics
    if True:
        df = unit_statistics(sbml_paths=sbml_paths)
        console.rule(style="white")
        console.print(df)
        console.rule(style="white")
    exit()

    # read the PETab problems
    import petab
    for yaml_path in yaml_paths:
        problem: petab.Problem = petab.Problem.from_yaml(yaml_path)
        console.print(problem)
        errors_exist = petab.lint.lint_problem(problem)
        console.print(f"PEtab errors: {errors_exist}")
        # print_problem(petab_problem=problem)

        # get units for SBML model
        if problem.sbml_model:
            model: libsbml.Model = problem.sbml_model
            doc: libsbml.SBMLDocument = model.getSBMLDocument()
            info = unit_statistics_for_doc(doc=doc)
            console.print(info)
            # TODO do the conversion to proper units



