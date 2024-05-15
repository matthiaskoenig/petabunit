"""Example of unit checking and conversion."""

from petabunit.console import console
from petabunit.sbmlunits import unit_statistics

examples = [
    "Elowitz_Nature2000/Elowitz_Nature2000.yaml",  # no units
    "simple_chain/simple_chain.yaml",  # complete example with units
]


def example_Elowitz2000() -> None:
    """Run Elowitz2000 example."""
    console.rule("Elowitz2000", style="white")


if __name__ == "__main__":
    from petabunit import EXAMPLES_DIR
    sbml_paths = [
        EXAMPLES_DIR / "Elowitz_Nature2000" / "model_Elowitz_Nature2000.xml",
        EXAMPLES_DIR / "enalapril_pbpk" / "enalapril_pbpk.xml",
        EXAMPLES_DIR / "simple_chain" / "simple_chain.xml",
        EXAMPLES_DIR / "simple_pk" / "simple_pk.xml",
    ]
    yaml_paths = [
        EXAMPLES_DIR / "Elowitz_Nature2000" / "Elowitz_Nature2000.yaml",
        EXAMPLES_DIR / "simple_chain" / "simple_chain.yaml",
    ]


    # unit statistics
    # df = unit_statistics(sbml_paths=sbml_paths)
    # console.rule(style="white")
    # console.print(df)
    # console.rule(style="white")

    # read the PETab problems
    import petab
    yaml_path = yaml_paths[1]
    problem: petab.Problem = petab.Problem.from_yaml(yaml_path)
    console.print(problem)
    errors_exist = petab.lint.lint_problem(problem)
    console.print(f"PEtab errors: {errors_exist}")

