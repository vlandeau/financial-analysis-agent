import numpy as np
import pandas as pd

from forecasting.features import TIME_TREND_COL_NAME


def create_quarter_features(
        df: pd.DataFrame,
        starting_value_for_time_trend: int = 0
) -> pd.DataFrame:
    """
    Create quarter-based features from a DataFrame with a 'date' index.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'date' column, with a quarter frequency.

    Returns:
        pd.DataFrame: DataFrame with additional date-based features.
    """
    df_with_features = df.copy()
    df_with_features[TIME_TREND_COL_NAME] = np.arange(starting_value_for_time_trend,
                                                      starting_value_for_time_trend + len(df_with_features))
    df_with_features["quarter"] = df_with_features.index.quarter
    seasonal_dummies = pd.get_dummies(df_with_features["quarter"], prefix="Q", drop_first=True)
    df_with_features = pd.concat([df_with_features, seasonal_dummies], axis=1)
    return df_with_features.astype(float)
