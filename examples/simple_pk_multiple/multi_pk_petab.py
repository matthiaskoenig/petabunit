from typing import Dict, List, Optional
from pathlib import Path
import roadrunner
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal, Covariance, gaussian_kde
from matplotlib import pyplot as plt

from petabunit import EXAMPLES_DIR
from petabunit.console import console
from petabunit.petabunits import MEASUREMENT_TIME_UNIT_COLUMN, MEASUREMENT_UNIT_COLUMN

FIG_PATH: Path = Path(__file__).parent / "results"

_LOG_2PI = np.log(2 * np.pi)

# Create class for distributions


class BivariateLogNormal:

    def __init__(self,
                 parameter_names: List[str],
                 mean: np.array,
                 cov: Covariance
                 ) -> None:
        self.parameter_names = parameter_names
        self.mean = mean
        self.cov = cov

    def rvs(self,
            size: int,
            seed: Optional[int]
            ) -> Dict[str, np.array]:

        if seed:
            np.random.seed(seed)

        multi_norm = multivariate_normal(self.mean, self.cov)
        sample = np.exp(multi_norm.rvs(size=size))

        result = {}
        for j, par in enumerate(self.parameter_names):
            result[par] = sample[:, j]

        return result

    def logpdf(self,
               x: np.array
               ) -> np.array:

        log_det_cov, rank = self.cov.log_pdet, self.cov.rank
        dev = np.log(x) - self.mean
        if dev.ndim > 1:
            log_det_cov = log_det_cov[..., np.newaxis]
            rank = rank[..., np.newaxis]
        maha = np.sum(np.square(self.cov.whiten(dev)) + 2 * np.log(x), axis=-1)

        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    def pdf(self,
            x: np.array
            ) -> np.array:
        return np.exp(self.logpdf(x))

    @staticmethod
    def plot_distributions(dsns, samples: List[Dict[str, np.array]]) -> None:

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        colors = ["tab:blue", "tab:red"]
        cmaps = ["Blues", "Reds"]

        def calculate_limits(samples):
            return xlims, ylims

        def plot_samples():
            for k, samples_data in enumerate(samples):
                df = pd.DataFrame.from_dict(samples_data)
                ax.plot(
                    df['kabs'], df['CL'],
                    'o',
                    color=colors[k],
                    markeredgecolor='k',
                    markersize=4,
                    alpha=0.8
                )

                # Gaussian KDE
                # xmin = df['kabs'].min()
                # xmax = df['kabs'].max()
                # ymin = df['CL'].min()
                # ymax = df['CL'].max()
                # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                # positions = np.vstack([X.ravel(), Y.ravel()])
                # values = np.vstack([df['kabs'], df['CL']])
                # kernel = gaussian_kde(values)
                # Z = np.reshape(kernel(positions).T, X.shape)
                # ax.imshow(np.rot90(Z), cmap=cmaps[k], extent=[xmin, xmax, ymin, ymax])

        plot_samples()

        # plot pdf
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

        for k, dsn in enumerate(dsns):
            xvec = np.linspace(start=xlims[0], stop=xlims[1], num=100)
            yvec = np.linspace(start=ylims[0], stop=ylims[1], num=100)
            x, y = np.meshgrid(xvec, yvec)

            xy = np.dstack((x, y))
            z = dsn.pdf(xy)

            cs = ax.contour(
                x, y, z,
                colors=colors[k],
                # cmap=cmaps[k],
                levels=20, alpha=1.0
            )

        # plot_samples()
        scale = "log"
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlabel('kabs')
        ax.set_ylabel('CL')


# Create class for simulation

class ODESimulation:

    def __init__(self,
                 model_path: Path,
                 samples: Dict[str, np.array],
                 compartment_starting_values: Dict[str, int]
                 ):
        self.model_path = model_path
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))

        integrator: roadrunner.Integrator = self.r.integrator
        integrator.setSetting("absolute_tolerance", 1e-6)
        integrator.setSetting("relative_tolerance", 1e-6)



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

        measurement_df.to_csv(self.model_path.parent / "measurements_multi_pk.tsv",
                              sep="\t", index=False)

        condition_df.to_csv(self.model_path.parent / "conditions_multi_pk.tsv",
                            sep="\t", index=False)

        parameter_df.to_csv(self.model_path.parent / "parameters_multi_pk.tsv",
                            sep='\t', index=False)

        observable_df.to_csv(self.model_path.parent / "observables_multi_pk.tsv",
                             sep='\t', index=False)


if __name__ == "__main__":
    seed = None # 1234
    n_samples = 100

    # sampling from distribution
    parameter_names = ['kabs', 'CL']

    # men
    mu_male = np.array([0.1, 0.5])  # mean in normal space
    cov_male = Covariance.from_diagonal([1, 1])
    dsn_male = BivariateLogNormal(mean=mu_male, cov=cov_male,
                                  parameter_names=parameter_names)
    samples_male = dsn_male.rvs(n_samples, seed=seed)
    console.rule("male", style="white")

    # women
    mu_female = np.array([10, 10])  # mean in normal space
    # mu_female = np.log(np.array([3, 3]))  # mean in normal space
    cov_female = Covariance.from_diagonal([1, 1])
    dsn_female = BivariateLogNormal(mean=mu_female, cov=cov_female,
                                    parameter_names=parameter_names)
    samples_female = dsn_female.rvs(n_samples, seed=seed)
    console.rule("female", style="white")

    # plot distributions
    BivariateLogNormal.plot_distributions(
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


    # 1. define distributions (multi-var log normal)
    # 2. calculate a PDF from that (PDF calculation) whatever analytical, complicated

    # 3. take samples
    # 4. do a kernel estimate of the samples => pdf (

