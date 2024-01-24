import pandas as pd

def calculate_correct_correspondences(df, mode="col"):
    """
    Calculate the proportion of correct correspondences in a DataFrame.

    Args:
        df: The DataFrame with UOT probabilities.
        mode: Either 'row' or 'col' to specify whether to calculate by row or column (default is 'row').

    Returns:
        float: The proportion of correct correspondences.
    """
    results = []
    if mode == "row":
        for source_type in df.index:
            target_type = df.loc[source_type].idxmax()
            probability = df.loc[source_type, target_type]
            results.append([source_type, target_type, probability])
    elif mode == "col":
        for source_type in df.columns:
            target_type = df[source_type].idxmax()
            probability = df.loc[target_type, source_type]
            results.append([source_type, target_type, probability])

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=['Source', 'Target', 'Probability'])

    # Calculate the proportion of correct correspondences
    correct_percentage = (results_df['Source'] == results_df['Target']).mean() * 100

    return correct_percentage
