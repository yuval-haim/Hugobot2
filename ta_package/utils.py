# ta_package/utils.py
import pandas as pd
import numpy as np
from .constants import ENTITY_ID, VALUE, TEMPORAL_PROPERTY_ID, TIMESTAMP
from scipy.stats import entropy

def assign_state(value, boundaries):
    """
    Given a value and a sorted list of boundaries, assign a state id (starting at 1).
    
    For example, with 3 bins (boundaries = [b1, b2]):
      if value < b1: state = 1
      if b1 <= value < b2: state = 2
      if value >= b2: state = 3
    """
    for i, b in enumerate(boundaries):
        if value < b:
            return i + 1
    return len(boundaries) + 1

def generate_candidate_cutpoints(df, nb_candidates):
    """
    Generate candidate cutpoints from the DataFrame's TemporalPropertyValue column.
    
    Parameters:
      df: A DataFrame that contains a column "TemporalPropertyValue".
      nb_candidates: Desired number of candidate cutpoints.
      
    Returns:
      A sorted list of candidate cutpoints.
    """
    values = df["TemporalPropertyValue"].dropna().unique()
    values = np.sort(values)
    # If there are fewer than 2 unique values, return an empty list.
    if len(values) < 2:
        return []
    # Evenly space candidate indices between 1 and len(values)-1:
    indices = np.linspace(1, len(values) - 1, num=nb_candidates, dtype=int)
    candidates = values[indices]
    return candidates.tolist()

def candidate_selection(df, nb_bins, scoring_function, nb_candidates=100):
    """
    Choose cutpoints from a pool of candidate cutpoints based on a scoring function.
    
    The candidate cutpoints are generated using generate_candidate_cutpoints().
    Then, iteratively, one candidate is chosen at a time to maximize the score.
    
    Parameters:
      df: A DataFrame with a column "TemporalPropertyValue".
      nb_bins: The final desired number of bins. (We choose nb_bins-1 cutpoints.)
      scoring_function: A function taking (df, cutoffs) and returning a numeric score.
      nb_candidates: Number of candidate cutpoints to generate initially.
      
    Returns:
      A tuple (chosen_cutpoints, chosen_scores) where:
       - chosen_cutpoints is the list of selected cutpoints (sorted), and
       - chosen_scores is a list of the corresponding scores.
    """
    # Generate candidate cutpoints.
    candidate_pool = generate_candidate_cutpoints(df, nb_candidates)
    chosen_cutpoints = np.array([], dtype=float)
    chosen_scores = np.array([], dtype=float)
    
    for _ in range(1, nb_bins):
        scores = np.full(len(candidate_pool), -np.inf)
        for i, candidate in enumerate(candidate_pool):
            # Skip candidate if it is already (or nearly) in the chosen_cutpoints.
            if len(chosen_cutpoints) > 0 and np.any(np.isclose(candidate, chosen_cutpoints)):
                continue
            
            # Create a list of suggested cutpoints.
            suggested = np.sort(np.append(chosen_cutpoints, candidate))
            bins_edges = [-np.inf] + suggested.tolist() + [np.inf]
            df_temp = df.copy()
            # Use pd.cut with duplicates dropped.
            try:
                df_temp = df_temp.assign(
                    Bin=pd.cut(df_temp["TemporalPropertyValue"], bins=bins_edges, labels=False, duplicates="drop")
                )
            except Exception as e:
                scores[i] = -np.inf
                continue
            
            scores[i] = scoring_function(df_temp, suggested.tolist())
        
        # If no valid candidate remains, break early.
        if len(scores) == 0 or np.all(np.isneginf(scores)):
            break
        
        best_idx = np.argmax(scores)
        best_candidate = candidate_pool[best_idx]
        chosen_cutpoints = np.append(chosen_cutpoints, best_candidate)
        chosen_scores = np.append(chosen_scores, scores[best_idx])
        # Remove this candidate from the pool.
        candidate_pool.pop(best_idx)
        # Also, remove any candidate that is nearly equal to a chosen candidate.
        candidate_pool = [c for c in candidate_pool if not np.any(np.isclose(c, chosen_cutpoints))]
        chosen_cutpoints = np.sort(chosen_cutpoints)
        chosen_scores = chosen_scores[np.argsort(chosen_cutpoints)]
    
    return list(chosen_cutpoints), list(chosen_scores)

def symmetric_kullback_leibler(p, q):
    if sum(p) == 0 or sum(q) == 0:
        return 0
    return 0.5 * (entropy(p, q) + entropy(q, p))

def paa_transform(
    data: pd.DataFrame,
    window_size: int = 3,
    agg_method: str = 'mean',
    timestamp_strategy: str = 'bin_left_normalized'
) -> pd.DataFrame:
    """
    Apply Piecewise Aggregate Approximation (PAA) to the time series.

    The input DataFrame should have columns:
      ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE.

    The function groups data by ENTITY_ID and TEMPORAL_PROPERTY_ID, partitions each group
    into non-overlapping windows of size `window_size`, and computes aggregated values.

    Parameters:
      data: pd.DataFrame
      window_size: int, window length.
      agg_method: str, one of 'mean', 'min', 'max'.
      timestamp_strategy: str, one of 'first', 'bin_left_normalized'.

    Returns:
      A new DataFrame with the aggregated time series.
    """
    agg_funcs = {'mean': np.mean, 'min': np.min, 'max': np.max}
    timestamp_strategies = ['first', 'bin_left_normalized']

    # Check parameters
    if agg_method not in agg_funcs:
        raise ValueError("agg_method must be 'mean', 'min', or 'max'.")
    if timestamp_strategy not in timestamp_strategies:
        raise ValueError(f"timestamp_strategy must be one of {timestamp_strategies}.")
    if window_size < 1:
        raise Exception('ERROR: Invalid window size parameter')
    if window_size == 1 or data.empty:
        return data

    def paa_group(group):
        group = group.sort_values(by=TIMESTAMP)
        n = len(group)
        rows = []
        for start in range(0, n, window_size):
            window = group.iloc[start:start + window_size]
            if window.empty:
                continue
            aggregated_value = agg_funcs[agg_method](window[VALUE])

            if timestamp_strategy == 'first':
                aggregated_time = window[TIMESTAMP].iloc[0]

            elif timestamp_strategy == 'bin_left_normalized':
                # Determine the bin for the window based on its first timestamp
                first_timestamp = window[TIMESTAMP].iloc[0]
                # Find the bin left edge
                bin_left = (first_timestamp // window_size) * window_size
                # Normalize by window size and shift to start from 1
                aggregated_time = (bin_left / window_size) + 1

            row = {
                ENTITY_ID: window[ENTITY_ID].iloc[0],
                TEMPORAL_PROPERTY_ID: window[TEMPORAL_PROPERTY_ID].iloc[0],
                TIMESTAMP: aggregated_time,
                VALUE: aggregated_value
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    df = data.groupby([ENTITY_ID, TEMPORAL_PROPERTY_ID], group_keys=False).apply(paa_group)
    df.loc[df[TEMPORAL_PROPERTY_ID] == -1, TIMESTAMP] = 0.0

    return df



def save_symbolic_series(symbolic_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the symbolic time series to a CSV file.
    Expected columns: EntityID, TemporalPropertyID, Timestamp, state.
    """
    symbolic_df.to_csv(output_path, index=False)

def save_states(states, output_path: str) -> None:
    """
    Save the computed states (cutoffs) to a CSV file.
    """
    pd.DataFrame([states]).to_csv(output_path, index=False)


def generate_KL_content(symbolic_series: pd.DataFrame, max_gap: int) -> str:
    """
    Generate the content for the KL file.
    
    The KL file format:
      - First line: "startToncepts"
      - Second line: "numberOfEntities,<number>"
      - Then, for each entity:
          ENTITY_ID;
          start_time,end_time,StateID,TemporalPropertyID;start_time,end_time,StateID,TemporalPropertyID;...
    
    Intervals for a given entity and property are merged if consecutive points share the same state and
    the gap between the current point and the previous point is less than or equal to max_gap.
    
    Parameters:
      symbolic_series (pd.DataFrame): DataFrame with columns: ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, state.
      max_gap (int): The maximum gap threshold for merging intervals.
      
    Returns:
      A string representing the contents of the KL file.
    """
    # Ensure the DataFrame is sorted.
    df = symbolic_series.sort_values(by=[ENTITY_ID, TIMESTAMP, TEMPORAL_PROPERTY_ID])
    kl_lines = []
    
    entities = df[ENTITY_ID].unique()
    kl_lines.append("startToncepts")
    kl_lines.append(f"numberOfEntities,{len(entities)}")
    
    # Process each entity individually.
    for entity in entities:
        entity_df = df[df[ENTITY_ID] == entity].sort_values(by=[TIMESTAMP, TEMPORAL_PROPERTY_ID])
        intervals = []
        # Group by TEMPORAL_PROPERTY_ID, then merge consecutive points into intervals.
        for tpid, group in entity_df.groupby(TEMPORAL_PROPERTY_ID):
            group = group.sort_values(by=TIMESTAMP)
            current_interval = None
            for _, row in group.iterrows():
                ts = row[TIMESTAMP]
                state = row["StateID"]
                if current_interval is None:
                    current_interval = {"start": ts, "end": ts, "StateID": state, TEMPORAL_PROPERTY_ID: tpid}
                else:
                    # If the same state and property, and the gap is within max_gap, extend the interval.
                    if state == current_interval["StateID"] and (ts - current_interval["end"]) <= max_gap:
                        current_interval["end"] = ts
                    else:
                        intervals.append(current_interval)
                        current_interval = {"start": ts, "end": ts, "StateID": state, TEMPORAL_PROPERTY_ID: tpid}
            if current_interval is not None:
                intervals.append(current_interval)
        # Sort intervals by start time and then by TEMPORAL_PROPERTY_ID.
        intervals = sorted(intervals, key=lambda x: (x["start"], x[TEMPORAL_PROPERTY_ID]))
        # Format intervals as "start_time,end_time,state,TEMPORAL_PROPERTY_ID"
        interval_strs = [f"{interval['start']},{interval['end']},{interval['StateID']},{interval[TEMPORAL_PROPERTY_ID]}" 
                         for interval in intervals]
        # Build the line for the entity.
        entity_line = f"{entity};\n" + ";".join(interval_strs) + ";"
        kl_lines.append(entity_line)
    
    return "\n".join(kl_lines)

def split_train_test(data: pd.DataFrame, train_ratio: float = 0.7):
        unique_ids = data[ENTITY_ID].unique()
        unique_ids = sorted(unique_ids)
        cutoff = int(len(unique_ids) * train_ratio)
        train_ids = unique_ids[:cutoff]
        test_ids = unique_ids[cutoff:]
        train = data[data[ENTITY_ID].isin(train_ids)]
        test = data[data[ENTITY_ID].isin(test_ids)]
        return train, test