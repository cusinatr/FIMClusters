"""Analysis of FIM data to find clusters of patients."""

from os import makedirs, path
import pandas as pd
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


###
# Paths and Parameters
###

file_name = "sample_data.xlsx"
sheet_name = "FIM total"
save_folder = "sample_Results"

n_weeks = 3
n_pats = None
remove_outliers = True

###
# Classes and functions
###


class FIMData:
    def __init__(self, file_name, sheet_name, save_folder="", n_weeks=3, n_pats=None):

        np.random.seed(0)

        self.file_name = file_name
        self.sheet_name = sheet_name
        self.save_folder = save_folder
        self.n_weeks = n_weeks
        self.n_pats = n_pats

        makedirs(save_folder, exist_ok=True)

        self.import_data()
        self.compute_delta()

    def import_data(self):
        """Import data and select patients."""

        self.data = pd.read_excel(
            self.file_name, sheet_name=self.sheet_name, index_col=0
        )
        self.n_pats = len(self.data) if self.n_pats is None else self.n_pats
        # Plot and save number of patients by week
        self.plot_pat_number()

        # Extract only columns of interest
        self.pat_codes = self.data.index.to_numpy()
        cols = ["T0"] + ["Week " + str(i + 1) for i in range(self.n_weeks)]
        self.data = self.data.loc[:, cols].copy()

        # Delete NaNs
        self.idx_nonan = ~self.data.isnull().any(axis=1)
        self.data = self.data[self.idx_nonan]

        # Plot distributions of raw data
        self.plot_distrib(
            self.data.to_numpy(), self.data.columns, fig_name="Distrib_raw"
        )

    def plot_pat_number(self):
        """Plot patients number by week"""

        cols_weeks = [c for c in self.data.columns if (c == "T0") or ("Week" in c)]
        n_per_week = len(self.data) - self.data.loc[:, cols_weeks].isna().sum()
        fig, ax = plt.subplots(1, 1, figsize=[6, 6])
        ax.plot(n_per_week.values, "k-o", markersize=10, lw=1)
        ax.axvline(x=3, ls="--", label="3 weeks")
        ax.set_xlabel("Week #", fontsize=15)
        ax.set_ylabel("Patients count", fontsize=15)
        ax.legend(fontsize=15)
        fig.savefig(path.join(self.save_folder, "Pats_per_week.png"), dpi=150)
        plt.close(fig)

    def plot_distrib(self, data, names, fig_name="Distrib"):
        """Plot distribution of features."""

        N_feat = data.shape[1]

        fig, axs = plt.subplots(
            nrows=1,
            ncols=N_feat,
            sharey=True,
            figsize=[6 * N_feat, 6],
            tight_layout=True,
        )
        for i, ax in enumerate(axs):
            ax.hist(data[:, i], bins=12)
            ax.set_xlabel(names[i], fontsize=12)
            if i == 0:
                ax.set_ylabel("Count", fontsize=12)
        plt.tight_layout()
        fig.savefig(path.join(self.save_folder, fig_name + ".png"), dpi=150)
        plt.close(fig)

    def compute_delta(self):
        """Compute delta feature between successive weeks."""

        self.X = self.data.to_numpy()[:, 1:] - self.data.to_numpy()[:, :-1]
        self.plot_distrib(
            self.X,
            [f"Week {i} - Week {i-1}" for i in range(1, self.n_weeks + 1)],
            fig_name="Distrib_diffs",
        )

    def outliers_detection(self, alpha=0.01):

        """
        Find outliers indices in a specified window of variables.
        Inputs:
            alpha: float
                Probability threshold to detect outliers.

        Outputs:
            outliers: list of outliers indices
            prob: probabilities associated with outliers
        """

        # Degres of freedom for Chi squared distribution
        dof = self.X.shape[1]

        # Mean
        mu = np.mean(self.X, axis=0).reshape(1, -1).astype(np.float64)

        # Covariance
        cov = np.cov(self.X, rowvar=False).astype(np.float64)

        # MD
        x_minus_mu = self.X - mu
        inv_covmat = np.linalg.inv(cov).astype(np.float64)
        mahal = x_minus_mu @ inv_covmat @ x_minus_mu.T
        MD = mahal.diagonal()

        # Check if covariance matrix is definite positive
        check_cov = np.linalg.cholesky(cov)

        # Flag as outlier
        self.outliers = []
        self.prob = []

        # Cut-off point
        C = chi2.ppf(
            (1 - alpha), df=dof
        )  # Mark as outlier if outside 1-alpha probability range
        for i, md in enumerate(MD):
            self.prob.append(chi2.cdf(md, df=dof))
            if md > C:
                self.outliers.append(i)
        self.prob = np.around((np.array(self.prob) * 100), 4)

        return self.outliers, self.prob

    def remove_outliers(self):
        self.X = np.delete(self.X, self.outliers, axis=0)

    def select_clusters(self):
        """Compute differentmetrics to select optimal number of clusters."""

        diffs = self.X.copy()

        n_clusters = list(range(2, 10))
        inertia = []
        silhouettes = []
        CHscore = []
        DBscore = []

        for n_clusts in n_clusters:
            kmeans = KMeans(n_clusters=n_clusts)
            kmeans.fit(diffs)
            y = kmeans.labels_
            inertia.append(kmeans.inertia_)
            silhouettes.append(metrics.silhouette_score(diffs, y, random_state=0))
            CHscore.append(metrics.calinski_harabasz_score(diffs, y))
            DBscore.append(metrics.davies_bouldin_score(diffs, y))

        # Plot
        fig, axs = plt.subplots(2, 2, figsize=[12, 12])
        axs[0, 0].plot(n_clusters, inertia, "-o")
        axs[0, 0].set_xlabel("# of clusters", fontsize=12)
        axs[0, 0].set_ylabel("Inertia", fontsize=12)
        axs[0, 1].plot(n_clusters, silhouettes, "-o")
        axs[0, 1].set_xlabel("# of clusters", fontsize=12)
        axs[0, 1].set_ylabel("Silhouette score", fontsize=12)
        axs[1, 0].plot(n_clusters, CHscore, "-o")
        axs[1, 0].set_xlabel("# of clusters", fontsize=12)
        axs[1, 0].set_ylabel("Calinski-Harabasz Index", fontsize=12)
        axs[1, 1].plot(n_clusters, DBscore, "-o")
        axs[1, 1].set_xlabel("# of clusters", fontsize=12)
        axs[1, 1].set_ylabel("Davies-Bouldin Index", fontsize=12)

        fig.savefig(path.join(self.save_folder, "Optim_clusters.png"), dpi=150)
        plt.close(fig)

    def clustering(self, n_clusters):
        """Perform the actual clustering"""

        self.n_clusters = n_clusters
        algo = KMeans(n_clusters=n_clusters)
        algo.fit(self.X)
        centers = algo.cluster_centers_

        #  Order labels by the center
        idx_sort = np.argsort(centers[:, 0])
        self.centers = centers[idx_sort]
        print("\nCenters for {} clusters:".format(n_clusters))
        print(centers)
        self.labels = np.array([np.where(idx_sort == l)[0][0] for l in algo.labels_])
        labels_prop = [
            np.sum(self.labels == l) / len(self.labels) for l in range(n_clusters)
        ]
        print_str = str(n_clusters) + " clusters:"
        for i in range(n_clusters):
            print_str += (
                " label "
                + str(i)
                + " = "
                + str(int(labels_prop[i] * self.X.shape[0]))
                + " patients;"
            )
        print(print_str)

        # Variances of clusters
        self.std = np.array(
            [np.std(self.X[self.labels == i], axis=0) for i in range(n_clusters)]
        )

        # Distance from clusters
        self.dist = algo.transform(self.X)[:, idx_sort]

        self.plot_clustering(
            [f"Week {i} - Week {i-1}" for i in range(1, self.n_weeks + 1)]
        )
        self.save_cluster_labels()
        self.save_cluster_distances()

    def plot_clustering(self, feat_names=None):
        """Plot clusters centers and variances along features."""

        colors = ["b", "r", "g", "m", "c", "orange"]
        fig, ax = plt.subplots(1, 1, figsize=[8, 8])
        x = np.array(range(self.X.shape[1]))
        for n in range(self.n_clusters):
            ax.plot(x, self.centers[n], "-o", lw=2.5, ms=10, color=colors[n])
            ax.fill_between(
                x,
                self.centers[n] - self.std[n],
                self.centers[n] + self.std[n],
                color=colors[n],
                alpha=0.2,
            )
        feat_names = (
            [str(i) for i in range(self.X.shape[1])]
            if feat_names is None
            else feat_names
        )
        ax.set_xticks(x)
        ax.set_xticklabels(feat_names, fontsize=12)
        ax.set_ylabel("Cluster mean", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(
            path.join(self.save_folder, f"{self.n_clusters}_clusters.png"), dpi=150
        )

        # Save another figure with "cumulative" differences
        fig, ax = plt.subplots(1, 1, figsize=[8, 8])
        x = np.array(range(self.X.shape[1] + 1))
        for n in range(self.n_clusters):
            new_centers = [0] + list(np.cumsum(self.centers[n]))
            new_std = [0] + [
                np.sqrt((self.std[: i + 1] ** 2).sum()) for i in range(len(self.std))
            ]
            ax.plot(x, new_centers, "-o", lw=2.5, ms=10, color=colors[n])
            ax.fill_between(
                x,
                np.array(new_centers) - np.array(new_std),
                np.array(new_centers) + np.array(new_std),
                color=colors[n],
                alpha=0.2,
            )
        feat_names = (
            [str(i) for i in range(self.X.shape[1])]
            if feat_names is None
            else feat_names
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["T0"] + [f"Week {i + 1}" for i in range(self.n_clusters)], fontsize=12
        )
        ax.set_ylabel("Cumulative Cluster mean", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(
            path.join(self.save_folder, f"{self.n_clusters}_clusters_cumul.png"),
            dpi=150,
        )

    def save_cluster_labels(self):
        """Save labels in an excel file."""

        idx_not_nans = np.where(self.idx_nonan)[0]
        idx_ok = [p for i, p in enumerate(idx_not_nans) if i not in self.outliers]

        # Save labels
        df_labs = pd.DataFrame(columns=["label"], index=self.pat_codes)
        df_labs.iloc[idx_ok, 0] = self.labels

        # Register outliers
        pat_outliers = self.pat_codes[idx_not_nans][self.outliers]
        df_labs.loc[df_labs.index.isin(pat_outliers)] = "outlier"

        # Save Dataframe
        df_labs.to_excel(path.join(self.save_folder, "cluster_labels.xlsx"))

    def save_cluster_distances(self):
        """Save distances from each cluster in an excel file."""

        idx_not_nans = np.where(self.idx_nonan)[0]
        idx_ok = [p for i, p in enumerate(idx_not_nans) if i not in self.outliers]

        # Save labels
        df_dist = pd.DataFrame(
            columns=[f"Cluster {i + 1}" for i in range(self.n_clusters)],
            index=self.pat_codes,
        )
        df_dist.iloc[idx_ok] = self.dist

        # Register outliers
        pat_outliers = self.pat_codes[idx_not_nans][self.outliers]
        df_dist.loc[df_dist.index.isin(pat_outliers)] = "outlier"

        # Save Dataframe
        df_dist.to_excel(path.join(self.save_folder, "cluster_distances.xlsx"))


if __name__ == "__main__":

    fim = FIMData(file_name, sheet_name, save_folder)
    fim.compute_delta()
    out, probs = fim.outliers_detection()
    if remove_outliers:
        fim.remove_outliers()
    fim.select_clusters()
    fim.clustering(3)
