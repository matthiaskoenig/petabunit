from dataclasses import dataclass
from typing import Callable, Dict, Union, List, Optional
from pathlib import Path
from matplotlib import pyplot as plt
import roadrunner
import pandas as pd
import numpy as np
import xarray as xr
from dataclasses import dataclass
from scipy.stats import rv_continuous, multivariate_normal, Covariance
from scipy.stats._multivariate import multivariate_normal_frozen
from matplotlib import pyplot as plt

from petabunit import EXAMPLES_DIR
from petabunit.console import console
from petabunit.petabunits import MEASUREMENT_TIME_UNIT_COLUMN, MEASUREMENT_UNIT_COLUMN

FIG_PATH: Path = Path(__file__).parent / "results"


# Create class for distributions

class BivariateLogNormal(multivariate_normal_frozen):
    """Based on scipy.stats.multivariate_normal"""

    def __init__(self,
                 parameter_names: List[str],
                 mu: np.array,
                 cov: Union[np.array, Covariance]):
        super(BivariateLogNormal, self).__init__(mean=mu, cov=cov)
        self.parameter_names = parameter_names

    def draw_sample(self, n: int) -> Dict[str, np.array]:
        sample = np.exp(self.rvs(n))
        result = {}
        for j, par in enumerate(self.parameter_names):
            result[par] = sample[:, j]
        return result

    def plot_dsn(self,
                 sample: Dict[str, np.array],
                 which: List[str] = None) -> None:
        # FIXME: Adjust the plot dimensions

        # Start with a square Figure.
        fig = plt.figure(figsize=(6, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal Axes and the main Axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        df = pd.DataFrame.from_dict(sample)
        x, y = np.mgrid[0:7:.1, 0:7:.1]
        xy = np.dstack((x, y))
        z = self.logpdf(xy)
        ax.contourf(x, y, z, cmap='coolwarm')
        ax.plot(df['kabs'], df['CL'], '*')
        ax.set_xlabel('K')
        ax.set_ylabel('CL')

        # ax_histx.tick_params(axis="x", labelbottom=False)
        # ax_histy.tick_params(axis="y", labelleft=False)
        #
        # binwidth = 0.25
        # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        # lim = (int(xymax / binwidth) + 1) * binwidth
        #
        # bins = np.arange(-lim, lim + binwidth, binwidth)
        # ax_histx.hist(df['kabs'], bins=bins)
        # ax_histy.hist(df['CL'], bins=bins, orientation='horizontal')


# Create class for simulation

class ODESimulation:

    def __init__(self,
                 model_path: Path,
                 samples: Dict[str, np.array],
                 compartment_starting_values: Dict[str, int]
                 ):
        self.model_path = model_path
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        self.samples = samples
        self.compartment_starting_values = compartment_starting_values

    def sim(self,
            sim_start: int = 0,
            sim_end: int = 10,
            sim_steps: int = 100,
            **kwargs):

        samples = pd.DataFrame.from_dict(self.samples)
        print(samples)
        dfs = []

        for _, row in samples.iterrows():
            for par_name, par in zip(row.index, row):
                self.r.setValue(par_name, par)

            s = self.r.simulate(start=sim_start,
                                end=sim_end,
                                steps=sim_steps,
                                **kwargs)
            df = pd.DataFrame(s, columns=s.colnames).set_index("time")
            dfs.append(df)

        dset = xr.concat([df.to_xarray() for df in dfs],
                         dim=pd.Index(np.arange(samples.shape[0]), name='sim'))

        return dset

    # def sim_plot(self,
    #              sim_df: xr.Dataset):
    #     pass
    #
    def to_petab(self,
                 sim_df: xr.Dataset):
        # FIXME: Add parameters [kabs, CL] and observable [all the y_s] table

        measurement_ls: List[pd.DataFrame] = []
        condition_ls: List[Dict[str, Optional[str, float, int]]] = []
        parameter_ls: List[Dict[str, Optional[str, float, int]]] = []
        observable_ls: List[Dict[str, Optional[str, float, int]]] = []

        for sim in sim_df['sim'].values:
            df_s = sim_df.isel(sim=sim).to_dataframe().reset_index()
            unique_measurement = []

            condition_ls.append({
                'conditionId': f'model1_data{sim}',
                'conditionName': ''
            })

            for col in ['y_gut', 'y_cent', 'y_peri']:
                if sim == sim_df['sim'].values[0]:
                    observable_ls.append({
                        'observableId': f'{col}_observable',
                        'observableFormula': col,
                        'observableName': col,
                        'noiseDistribution': 'normal',
                        'noiseFormula': 1,
                        'observableTransformation': 'lin',
                        'observableUnit': 'mmol/l'
                    })

                condition_ls[-1].update({col: self.compartment_starting_values[col]})
                col_brackets = '[' + col + ']'
                for k, row in df_s.iterrows():
                    unique_measurement.append({
                        "observableId": f"{col}_observable",
                        "preequilibrationConditionId": None,
                        "simulationConditionId": f"model1_data{sim}",
                        "measurement": row[col_brackets], # !
                        MEASUREMENT_UNIT_COLUMN: "mmole/l",
                        "time": row["time"], # !
                        MEASUREMENT_TIME_UNIT_COLUMN: "second",
                        "observableParameters": None,
                        "noiseParameters": None,
                    })

            measurement_sim_df = pd.DataFrame(unique_measurement)

            measurement_ls.append(measurement_sim_df)

        parameters: List[str] = list(self.samples.keys())

        for par in parameters:
            parameter_ls.append({
                'parameterId': par,
                'parameterName': par,
                'parameterScale': 'log10',
                'lowerBound': 0.01,
                'upperBound': 100,
                'nominalValue': 1,
                'estimate': 1,
                'parameterUnit': 'l/min'
            })

        measurement_df = pd.concat(measurement_ls)
        condition_df = pd.DataFrame(condition_ls)
        parameter_df = pd.DataFrame(parameter_ls)
        observable_df = pd.DataFrame(observable_ls)
        # console.print(measurement_df.info())
        # console.print(measurement_df.groupby(['simulationConditionId']).size())

        measurement_df.to_csv(self.model_path.parent / "measurements_multi_pk.tsv",
                              sep="\t", index=False)

        condition_df.to_csv(self.model_path.parent / "conditions_multi_pk.tsv",
                            sep="\t", index=False)

        parameter_df.to_csv(self.model_path.parent / "parameters_multi_pk.tsv",
                            sep='\t', index=False)

        observable_df.to_csv(self.model_path.parent / "observables_multi_pk.tsv",
                             sep='\t', index=False)




# Create class for petab format


if __name__ == "__main__":
    mu = np.array([0, 0])
    cov = Covariance.from_diagonal([1, 1])
    MODEL_PATH: Path = Path(__file__).parent / "simple_pk.xml"
    true_distribution = BivariateLogNormal(mu=mu, cov=cov, parameter_names=['kabs', 'CL'])
    true_samples = true_distribution.draw_sample(5)
    compartment_starting_values = {'y_gut': 1, 'y_cent': 0, 'y_peri': 0}

    true_distribution.plot_dsn(true_samples)
    plt.savefig(str(FIG_PATH) + '/00_dsn.png')

    ode_sim = ODESimulation(model_path=MODEL_PATH, samples=true_samples,
                            compartment_starting_values=compartment_starting_values)
    synth_df = ode_sim.sim()
    # console.print(synth_df)
    ode_sim.to_petab(synth_df)


