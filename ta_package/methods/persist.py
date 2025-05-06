# File: ta_package/methods/persist.py
import numpy as np
import pandas as pd
from collections import Counter
from .base import TAMethod
from ..utils import assign_state, candidate_selection, symmetric_kullback_leibler
from ..constants import TEMPORAL_PROPERTY_ID, VALUE

class Persist(TAMethod):
    def __init__(self, bins: int, per_variable: bool = True):
        """
        Parameters:
            bins (int): Number of bins desired.
            per_variable (bool): If True, process each variable separately.
        """
        self.bins = bins
        self.per_variable = per_variable
        self.boundaries = None

    @staticmethod
    def _marginal_probabilities(discrete_vals, nb_bins):
        marginal_probs = np.zeros((nb_bins, nb_bins))
        if len(discrete_vals) < 2:
            return marginal_probs
        transitions = discrete_vals[1:]
        for prev, cur in zip(discrete_vals[:-1], transitions):
            marginal_probs[int(prev), int(cur)] += 1
        marginal_probs = marginal_probs / (len(discrete_vals) - 1)
        return marginal_probs

    @staticmethod
    def _state_probabilities(discrete_vals, nb_bins):
        c = Counter(discrete_vals)
        probs = np.array([c.get(i, 0) for i in range(nb_bins)])
        if probs.sum() == 0:
            return probs
        return probs / float(probs.sum())

    @staticmethod
    def _all_states_persistence(discrete_vals, nb_bins):
        # Calculate state probabilities.
        state_probs = Persist._state_probabilities(discrete_vals, nb_bins)
        # Calculate marginal probabilities.
        marginal_probs = Persist._marginal_probabilities(discrete_vals, nb_bins)
        # For each state, compute a score.
        scores = []
        for i in range(nb_bins):
            m = marginal_probs[i, i]  # Simplified: probability of self-transition
            s = state_probs[i]
            # Use symmetric KL divergence between [m, 1-m] and [s, 1-s]
            scores.append(symmetric_kullback_leibler([m, 1-m], [s, 1-s]))
        if np.any(np.isinf(scores)):
            return np.inf
        return np.mean(scores)

    def _generate_cutpoints(self, df: pd.DataFrame):
        """
        Use candidate_selection to choose cutpoints.
        The scoring function returns the persistence score calculated over all states.
        """
        candidates, scores = candidate_selection(
            df,
            self.bins,
            lambda d, cutoffs: Persist._all_states_persistence(d[VALUE].values, len(cutoffs) + 1)
        )
        return candidates

    def fit(self, data: pd.DataFrame) -> None:
        if self.per_variable:
            boundaries = {}
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                boundaries[tpid] = self._generate_cutpoints(group)
            self.boundaries = boundaries
        else:
            self.boundaries = self._generate_cutpoints(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if self.per_variable:
            data["state"] = data.apply(lambda row: assign_state(row[VALUE], self.boundaries.get(row[TEMPORAL_PROPERTY_ID])), axis=1)
        else:
            data["state"] = data[VALUE].apply(lambda v: assign_state(v, self.boundaries))
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        return self.boundaries

def persist(data: pd.DataFrame, bins: int, per_variable: bool = True):
    method_instance = Persist(bins, per_variable)
    symbolic_series = method_instance.fit_transform(data)
    states = method_instance.get_states()
    return symbolic_series, states
