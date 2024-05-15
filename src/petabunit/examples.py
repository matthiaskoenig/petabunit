"""Example of unit checking and conversion."""

from petabunit.console import console

examples = [
    "Elowitz_Nature2000/Elowitz_Nature2000.yaml",  # no units
    "simple_chain/simple_chain.yaml",  # complete example with units

]

def example_Elowitz2000() -> None:
    """Run Elowitz2000 example."""
    console.rule("Elowitz2000", style="white")


def example_Elowitz2000() -> None:
    """Run Elowitz2000 example."""
    console.rule("Elowitz2000", style="white")


if __name__ == "__main__":
    example_Elowitz2000()
    example_Elowitz2000_Units()
