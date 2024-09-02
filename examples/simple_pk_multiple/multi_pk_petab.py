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


# FIXME: add a new child class for logmultinorm from multivariate_normal_gen.
#   Use the method _logpdf and calculate the log of the lognorm
#   Done for computational issues
class BivariateNormal(multivariate_normal_frozen):
    """Based on scipy.stats.multivariate_normal"""

    def __init__(self,
                 parameter_names: List[str],
                 mu: np.array,
                 cov: Union[np.array, Covariance]):
        super(BivariateNormal, self).__init__(mean=mu, cov=cov)
        self.parameter_names = parameter_names
        self.mu_data = mu
        self.cov_data = cov

    def draw_sample(self, n: int, seed: Optional[int] = None) -> Dict[str, np.array]:
        """Draws samples from the distribution."""
        if seed:
            np.random.seed(seed)
        sample = np.exp(self.rvs(n))
        result = {}
        for j, par in enumerate(self.parameter_names):
            result[par] = sample[:, j]
        return result

    @staticmethod
    def plot_distributions(dsns, samples: List[Dict[str, np.array]]) -> None:


        # Start with a square Figure.
        # fig = plt.figure(figsize=(6, 6))
        # # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # # the size of the marginal Axes and the main Axes in both directions.
        # # Also adjust the subplot parameters for a square plot.
        # gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        #                       left=0.1, right=0.9, bottom=0.1, top=0.9,
        #                       wspace=0.05, hspace=0.05)
        # # Create the Axes.
        # ax = fig.add_subplot(gs[1, 0])
        # ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        colors = ["tab:blue", "tab:red"]
        cmaps = ["Blues", "Reds"]

        # plot distribution
        # flim = 3
        # xmin = -max(1.0, self.mu_data[0]) * flim
        # xmax = max(1.0, self.mu_data[0]) * flim
        # ymin = -max(1.0, self.mu_data[1]) * flim
        # ymax = max(1.0, self.mu_data[1]) * flim
        # console.print(xmax, ymax)



        # plot samples
        for k, samples_data in enumerate(samples):
            df = pd.DataFrame.from_dict(samples_data)
            ax.plot(
                df['kabs'], df['CL'],
                '*',
                color=colors[k],
                markeredgecolor='k',
                markersize=10,
            )

        # plot pdf
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        for k, dsn in enumerate(dsns):
            xvec = np.linspace(start=xlims[0], stop=xlims[1], num=100)
            yvec = np.linspace(start=ylims[0], stop=ylims[1], num=100)
            x, y = np.meshgrid(xvec, yvec)

            xy = np.dstack((x, y))
            z = np.exp(dsn.pdf(xy))  # FIXME: Change to lognormal pdf

            # z[z<0.1] = np.NaN  # filter low probabilities
            cs = ax.contour(x, y, z, cmap=cmaps[k], levels=100)
            # fig.colorbar(cs, label="pdf")

        ax.set_xlabel('kabs')
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
            **kwargs) -> xr.Dataset:

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


if __name__ == "__main__":
    seed = None  # 1234
    n_samples = 50

    # sampling from distribution
    parameter_names = ['kabs', 'CL']

    # men
    mu_male = np.log(np.array([0.5, 0.5]))  # mean in normal space
    cov_male = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    dsn_male = BivariateNormal(mu=mu_male, cov=cov_male, parameter_names=parameter_names)
    samples_male = dsn_male.draw_sample(n_samples, seed=seed)
    console.rule("male", style="white")
    console.print(samples_male)

    # women
    mu_female = np.log(np.array([2, 2]))  # mean in normal space
    cov_female = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    dsn_female = BivariateNormal(mu=mu_female, cov=cov_female, parameter_names=parameter_names)
    samples_female = dsn_female.draw_sample(n_samples, seed=seed)
    console.rule("female", style="white")
    console.print(samples_female)

    # plot distributions
    BivariateNormal.plot_distributions(
        dsns=[dsn_male, dsn_female],
        samples=[samples_male, samples_female],
    )
    plt.savefig(str(FIG_PATH) + '/00_dsn.png')
    plt.show()


    # simulation
    MODEL_PATH: Path = Path(__file__).parent / "simple_pk.xml"
    compartment_starting_values = {'y_gut': 1, 'y_cent': 0, 'y_peri': 0}
    ode_sim = ODESimulation(model_path=MODEL_PATH, samples=samples_male,
                            compartment_starting_values=compartment_starting_values)
    synth_dset: xr.Dataset = ode_sim.sim()
    console.print(synth_dset)

    # convert to PeTab problem
    ode_sim.to_petab(synth_dset)


