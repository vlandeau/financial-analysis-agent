import pandas as pd
from forecasting.feature_engineering import create_quarter_features


def test_create_quarter_features_without_starting_value_for_time_trend():
    # Given
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2020-01-01", periods=8, freq="Q"),
            "financial_data_value": [100, 110, 120, 130, 140, 150, 160, 170],
        }
    )

    df.set_index("date", inplace=True)

    # When
    df_with_features = create_quarter_features(df)

    # Then
    expected_df_with_features = pd.DataFrame(
        {
            "financial_data_value": [100, 110, 120, 130, 140, 150, 160, 170],
            "time_trend": [0, 1, 2, 3, 4, 5, 6, 7],
            "quarter": [1, 2, 3, 4, 1, 2, 3, 4],
            "Q_2": [0, 1, 0, 0, 0, 1, 0, 0],
            "Q_3": [0, 0, 1, 0, 0, 0, 1, 0],
            "Q_4": [0, 0, 0, 1, 0, 0, 0, 1],
        },
        index=pd.date_range(start="2020-01-01", periods=8, freq="Q"),
    )
    expected_df_with_features.index.name = "date"

    for col in expected_df_with_features.columns:
        assert col in df_with_features.columns, f"Missing column: {col}"

    pd.testing.assert_frame_equal(
        df_with_features[expected_df_with_features.columns],
        expected_df_with_features,
        check_dtype=False,
        check_freq=False,
    )


def test_create_quarter_features_with_starting_value_for_time_trend():
    # Given
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2020-01-01", periods=4, freq="Q"),
            "financial_data_value": [100, 110, 120, 130],
        }
    )

    df.set_index("date", inplace=True)

    starting_value_for_time_trend = 10

    # When
    df_with_features = create_quarter_features(df, starting_value_for_time_trend)

    # Then
    expected_time_trend = [10, 11, 12, 13]
    assert all(df_with_features["time_trend"] == expected_time_trend), (
        "Time trend values do not match expected values"
    )
