import numpy as np
import pandas as pd
from .base import TAMethod
from ..utils import assign_state, paa_transform
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

class Gradient(TAMethod):
    def __init__(self, gradient_window_size: int, method: str = 'quantile',
                 bins: int = 3, close_to_zero_percentage: float = 30.0,
                 knowledge_cutoffs: list = None, per_variable: bool = True,
                 paa_method: str = None, paa_window: int = None):
        """
        Parameters:
          gradient_window_size (int): Window size for computing the gradient.
          method (str): 'quantile' uses data-driven boundaries; 'knowledge' uses predefined cutoffs.
          bins (int): Number of bins (typically 3) used when method is 'quantile'.
          close_to_zero_percentage (float): Percentage of samples to fall into the middle bin (only for quantile method).
          knowledge_cutoffs (list): A list of cutoff values for the 'knowledge'-based method.
                                    (For example, [-90, 0, 90]).  
          per_variable (bool): Process each TemporalPropertyID separately.
          paa_method (str): Optional PAA method; default is None (no PAA applied).
          paa_window (int): Optional PAA window size; default is None.
        """
        self.gradient_window_size = gradient_window_size
        self.method = method
        self.bins = bins
        self.close_to_zero_percentage = close_to_zero_percentage
        self.knowledge_cutoffs = knowledge_cutoffs
        self.per_variable = per_variable
        self.boundaries = None
        self.paa_method = paa_method
        self.paa_window = paa_window

    def _compute_angles_for_group(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(by=TIMESTAMP).copy()
        def compute_angle(row):
            start_time = row[TIMESTAMP] - self.gradient_window_size
            window = group[(group[TIMESTAMP] >= start_time) & (group[TIMESTAMP] <= row[TIMESTAMP])]
            if len(window) < 2:
                return np.nan  # Not enough points to compute slope.
            x = window[TIMESTAMP].values
            y = window[VALUE].values
            slope, _ = np.polyfit(x, y, 1)
            # Convert the slope to an angle in degrees.
            angle = np.degrees(np.arctan(slope))
            return angle
        group["angle"] = group.apply(compute_angle, axis=1)
        return group

    def _compute_angles(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.groupby([ENTITY_ID, TEMPORAL_PROPERTY_ID], group_keys=False).apply(self._compute_angles_for_group)

    def _determine_boundaries_quantile(self, angles: pd.Series) -> list:
        central_fraction = self.close_to_zero_percentage / 100.0
        tail_fraction = (1 - central_fraction) / 2
        lower_boundary = angles.quantile(tail_fraction)
        upper_boundary = angles.quantile(1 - tail_fraction)
        return [lower_boundary, upper_boundary]

    def fit(self, data: pd.DataFrame) -> None:
        # Apply PAA preprocessing if specified.
        if self.paa_method is not None and self.paa_window is not None:
            data = paa_transform(data, self.paa_window, agg_method=self.paa_method)
        computed = self._compute_angles(data)
        if self.per_variable:
            self.boundaries = {}
            for tpid, group in computed.groupby(TEMPORAL_PROPERTY_ID):
                angles = group["angle"].dropna()
                if self.method == 'quantile':
                    self.boundaries[tpid] = self._determine_boundaries_quantile(angles)
                elif self.method == 'knowledge':
                    if not self.knowledge_cutoffs or len(self.knowledge_cutoffs) != 3:
                        raise ValueError("Knowledge-based method requires a list of three cutoff values.")
                    # Here we use the full list of cutoffs as supplied.
                    self.boundaries[tpid] = self.knowledge_cutoffs
                else:
                    raise ValueError(f"Unknown method: {self.method}")
        else:
            if self.method == 'quantile':
                self.boundaries = self._determine_boundaries_quantile(computed["angle"].dropna())
            elif self.method == 'knowledge':
                if not self.knowledge_cutoffs or len(self.knowledge_cutoffs) != 3:
                    raise ValueError("Knowledge-based method requires a list of three cutoff values.")
                self.boundaries = self.knowledge_cutoffs
            else:
                raise ValueError(f"Unknown method: {self.method}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        # Apply PAA preprocessing if specified.
        if self.paa_method is not None and self.paa_window is not None:
            data = paa_transform(data, self.paa_window, agg_method=self.paa_method)
        computed = self._compute_angles(data)
        if self.per_variable:
            computed["state"] = computed.apply(
                lambda row: assign_state(row["angle"], self.boundaries.get(row[TEMPORAL_PROPERTY_ID], [])),
                axis=1
            )
        else:
            default_boundaries = self.boundaries if isinstance(self.boundaries, list) else []
            computed["state"] = computed["angle"].apply(
                lambda a: assign_state(a, default_boundaries)
            )
        return computed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        return self.boundaries

def gradient(data: pd.DataFrame, gradient_window_size: int, **kwargs):
    # Optionally, drop rows with -1 in TEMPORAL_PROPERTY_ID if needed.
    data = data[data[TEMPORAL_PROPERTY_ID] != -1]
    grad = Gradient(gradient_window_size, **kwargs)
    symbolic_series = grad.fit_transform(data)
    states = grad.get_states()
    return symbolic_series, states
