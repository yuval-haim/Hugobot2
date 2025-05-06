import pandas as pd
from .base import TAMethod
from ..utils import assign_state
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

class KnowledgeBased(TAMethod):
    def __init__(self, states: dict, per_variable: bool = True):
        """
        Parameters:
          states (dict): A dictionary of pre-determined boundaries (cutoffs) for each variable.
                         Format example: { variable_id: [cutoff1, cutoff2, ...],
                                             "default": [default_cutoff1, default_cutoff2, ...] }
          per_variable (bool): If True, each TemporalPropertyID is processed separately (default True).
        """
        self.states = states
        self.per_variable = per_variable

    def fit(self, data: pd.DataFrame) -> None:
        """
        In the knowledge-based method no learning is done—the boundaries are provided by the user.
        """
        self.boundaries = self.states

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Using the provided boundaries (self.boundaries), assign a state id for each sample.
        If a specific variable is not found in the dictionary, the method falls back to the default boundaries.
        """
        data = data.copy()
        if self.per_variable:
            # For each row, retrieve the boundaries for its variable;
            # if not found, use the boundaries under the key "default" (or an empty list if neither exists).
            data["state"] = data.apply(
                lambda row: assign_state(
                    row[VALUE],
                    self.boundaries.get(row[TEMPORAL_PROPERTY_ID], self.boundaries.get("default", []))
                ),
                axis=1
            )
        else:
            default_boundaries = self.boundaries.get("default", [])
            data["state"] = data[VALUE].apply(
                lambda v: assign_state(v, default_boundaries)
            )
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        return self.boundaries

# Method-level convenience function.
def knowledge(data: pd.DataFrame, states: dict, per_variable: bool = True):
    kb = KnowledgeBased(states, per_variable=per_variable)
    symbolic_series = kb.fit_transform(data)
    return symbolic_series, kb.get_states()
