from pathlib import Path
from matplotlib import pyplot as plt
import roadrunner
import pandas as pd
import numpy as np

from petabunit import EXAMPLES_DIR
from petabunit.console import console
from petabunit.petabunits import MEASUREMENT_TIME_UNIT_COLUMN, MEASUREMENT_UNIT_COLUMN


def example_simulation(model_path: Path) -> pd.DataFrame:
    np.random.seed(1234)
    r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
    s = r.simulate(start=0, end=5, steps=10)
    df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)
    console.print(df)

    f, ax = plt.subplots()
    ax.plot(df.time, df["[S1]"], color="tab:blue")
    ax.plot(df.time, df["[S2]"], color="tab:orange")

    # create measurement data
    # add random values;
    mu = 0
    sigma = 0.1
    df["S1_data"] = df["[S1]"] + np.random.normal(mu, sigma, len(df))
    df["S2_data"] = df["[S2]"] + np.random.normal(mu, sigma, len(df))
    plot_kwargs = {
        "linestyle": "",
        "marker": "o",
        "markeredgecolor": "black",
    }
    ax.plot(df.time, df["S1_data"], color="tab:blue", **plot_kwargs)
    ax.plot(df.time, df["S2_data"], color="tab:orange", **plot_kwargs)
    ax.set_xlabel("time")
    ax.set_ylabel("concentration")
    plt.show()
    return df


def create_petab_example(model_path: Path):
    df = example_simulation(model_path)

    # measurementData
    data = []
    for k, row in df.iterrows():
        data.append({
            "observableId": "S1_observable",
            "preequilibrationConditionId": None,
            "simulationConditionId": "model1_data1",
            "measurement": 	row["S1_data"],
            MEASUREMENT_UNIT_COLUMN: "mmole/l",
            "time": row["time"],
            MEASUREMENT_TIME_UNIT_COLUMN: "second",
            "observableParameters": None,
            "noiseParameters": None,
        })
        data.append({
            "observableId": "S2_observable",
            "preequilibrationConditionId": None,
            "simulationConditionId": "model1_data1",
            "measurement": 	row["S2_data"],
            MEASUREMENT_UNIT_COLUMN: "mmole/l",
            "time": row["time"],
            MEASUREMENT_TIME_UNIT_COLUMN: "second",
            "observableParameters": None,
            "noiseParameters": None,
        })
    measurement_df = pd.DataFrame(data)
    measurement_df.to_csv(model_path.parent / "measurements_simple_chain.tsv", sep="\t", index=False)


if __name__ == "__main__":
    model_path: Path = Path(__file__).parent / "simple_chain.xml"
    # example_simulation(model_path)
    create_petab_example(model_path)

    # check petab
    import petab
    yaml_path = EXAMPLES_DIR / "simple_chain" / "simple_chain.yaml"
    problem: petab.Problem = petab.Problem.from_yaml(yaml_path)
    console.print(problem)
    errors_exist = petab.lint.lint_problem(problem)
    console.print(f"PEtab errors: {errors_exist}")

