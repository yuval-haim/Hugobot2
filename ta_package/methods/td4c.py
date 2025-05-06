# File: ta_package/methods/td4c.py
import numpy as np
import pandas as pd
from scipy.stats import entropy
from .base import TAMethod
from ..utils import assign_state, candidate_selection, symmetric_kullback_leibler
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

class TD4C(TAMethod):
    def __init__(self, bins: int, per_variable: bool = True, distance_measure: str = "kullback_leibler"):
        """
        Parameters:
            bins (int): Desired number of bins (discretization intervals).
            per_variable (bool): If True, each TemporalPropertyID is fitted independently.
            distance_measure (str): Determines which distance function to use. Options:
                - "kullback_leibler": Use symmetric Kullback–Leibler divergence.
                - "entropy": Use the absolute difference of entropies.
                - "cosine": Use cosine similarity.
        """
        self.bins = bins
        self.per_variable = per_variable
        self.boundaries = None
        if distance_measure == "kullback_leibler":
            self._distance_measure = symmetric_kullback_leibler
        elif distance_measure == "entropy":
            self._distance_measure = lambda p, q: abs(entropy(p) - entropy(q))
        elif distance_measure == "cosine":
            self._distance_measure = lambda p, q: np.dot(p, q) / np.sqrt(np.dot(p, p) * np.dot(q, q))
        else:
            raise ValueError("Unsupported distance measure: " + distance_measure)

    def _generate_cutpoints(self, df: pd.DataFrame):
        """
        For a given DataFrame (corresponding to one variable), choose candidate cutpoints via candidate_selection.
        The scoring function compares class distributions across bins using the chosen distance measure.
        """
        # Ensure class information exists; if not, create a default class (e.g., 0).
        if 'Class' not in df.columns:
            df = df.assign(Class=0)
        # candidate_selection returns (candidates, scores); we only use the candidates.
        candidates, scores = candidate_selection(
            df,
            self.bins,
            lambda d, cutoffs: self._ddm_scoring_function(d, cutoffs)
        )
        return candidates

    def _ddm_scoring_function(self, df: pd.DataFrame, cutoffs):
        """
        Given a DataFrame and a list of cutoffs, compute a score.
        The score is calculated by first discretizing df[VALUE] based on the cutoffs,
        then for each class (from df['Class']) computing the distribution over bins and finally
        summing pairwise distances between the class distributions.
        """
        bins_array = [-np.inf] + list(cutoffs) + [np.inf]
        df = df.assign(Bin=pd.cut(df[VALUE], bins=bins_array, labels=False))
        classes = sorted(df['Class'].unique())
        nb_bins = len(bins_array) - 1
        class_distribs = np.zeros((len(classes), nb_bins))
        for i, cls in enumerate(classes):
            sub = df[df['Class'] == cls]
            if sub.empty:
                continue
            counts = sub['Bin'].value_counts().sort_index().values
            if counts.sum() > 0:
                # Build a probability vector of length nb_bins.
                v = np.zeros(nb_bins)
                v[:len(counts)] = counts
                class_distribs[i] = v / v.sum()
        score = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                score += self._distance_measure(class_distribs[i], class_distribs[j])
        return score

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the TD4C model by generating cutpoints for each variable.
        In per_variable mode, fit each TemporalPropertyID independently.
        """
        if self.per_variable:
            boundaries = {}
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                # If a mapping of entity classes exists, merge it in.
                if 'Class' not in group.columns and hasattr(self, 'entity_class') and self.entity_class:
                    group = group.assign(Class=group[ENTITY_ID].map(self.entity_class))
                else:
                    group = group.assign(Class=0)
                boundaries[tpid] = self._generate_cutpoints(group)
            self.boundaries = boundaries
        else:
            if 'Class' not in data.columns and hasattr(self, 'entity_class') and self.entity_class:
                data = data.assign(Class=data[ENTITY_ID].map(self.entity_class))
            else:
                data = data.assign(Class=0)
            self.boundaries = self._generate_cutpoints(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the learned cutpoints.
        For each sample, assign a state via the common assign_state() helper.
        """
        data = data.copy()
        if self.per_variable:
            data["state"] = data.apply(
                lambda row: assign_state(row[VALUE], self.boundaries.get(row[TEMPORAL_PROPERTY_ID], [])),
                axis=1
            )
        else:
            data["state"] = data[VALUE].apply(
                lambda v: assign_state(v, self.boundaries if self.boundaries is not None else [])
            )
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        """Return the computed boundaries."""
        return self.boundaries

def td4c(data: pd.DataFrame, bins: int, per_variable: bool = True, distance_measure: str = "kullback_leibler"):
    """
    Convenience function to run TD4C on a dataset.
    Parameters:
      data: Input DataFrame.
      bins: Number of bins desired.
      per_variable: Whether to fit each variable separately.
      distance_measure: Which distance measure to use ("kullback_leibler", "entropy", or "cosine").
    Returns:
      symbolic_series: Transformed DataFrame with a "state" column (local state id).
      states: The boundaries (cutpoints) computed per variable.
    """
    method_instance = TD4C(bins, per_variable, distance_measure=distance_measure)
    symbolic_series = method_instance.fit_transform(data)
    states = method_instance.get_states()
    return symbolic_series, states
