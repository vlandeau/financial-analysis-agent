import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import statsmodels.api as sm

from forecasting.feature_engineering import create_quarter_features
from forecasting.features import FEATURE_LIST


class TimeSeriesDataPoint(BaseModel):
    """Represents a single data point in the historical time series for a given financial metric."""

    date: str = Field(
        ..., description="The date of the data point in 'YYYY-MM-DD' format."
    )
    financial_data_value: float = Field(
        ..., description="The financial data value for that period."
    )


class TrainAndForecastInput(BaseModel):
    """Input schema for the dynamic training and forecasting tool."""

    historical_data: List[TimeSeriesDataPoint] = Field(
        ...,
        description="A list of historical data points, each with a date and revenue.",
    )
    num_quarters_to_forecast: int = Field(
        ...,
        gt=0,  # Must be greater than 0
        description="The number of future quarters to forecast after training on the historical data.",
    )
    confidence_interval: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="The confidence level for the prediction intervals (default is 0.95).",
    )


class PredictionIntervalDataPoint(BaseModel):
    """Represents a single forecasted data point with prediction intervals."""

    date: str = Field(
        ..., description="The date of the forecasted data point in 'YYYY-MM-DD' format."
    )
    financial_data_likeliest_value: float = Field(
        ..., description="The forecasted financial data value for that period."
    )
    financial_data_lower_bound: float = Field(
        ..., description="The lower bound of the prediction interval."
    )
    financial_data_upper_bound: float = Field(
        ..., description="The upper bound of the prediction interval."
    )


@tool(args_schema=TrainAndForecastInput)
def forecast_future_financial_data(
    historical_data: List[TimeSeriesDataPoint],
    num_quarters_to_forecast: int,
    confidence_interval: float = 0.95,
) -> List[PredictionIntervalDataPoint]:
    """
    Trains a simple forecasting model on historical financial data and predicts future values.

    Args:
        historical_data (List[TimeSeriesDataPoint]): A list of historical data points, each with a date and financial data value.
        num_quarters_to_forecast (int): The number of future quarters to forecast after training on the historical data.
        confidence_interval (float): The confidence level for the prediction intervals (default is 0.95).

    Returns:
        List[TimeSeriesDataPoint]: A list of forecasted data points for the specified number of future quarters.
    """
    df = _create_dataframe_indexed_by_quarter(historical_data)
    df_with_quarter_features = create_quarter_features(df)
    df_with_constant = sm.add_constant(df_with_quarter_features, has_constant="add")

    target_col = "target"
    data_value_col = "financial_data_value"
    is_financial_data_positive = (df[data_value_col] > 0).all()
    if is_financial_data_positive:
        df_with_constant[target_col] = np.log(df_with_constant[data_value_col])
    else:
        df_with_constant[target_col] = df_with_constant[data_value_col]

    model = sm.OLS(df_with_constant[target_col], df_with_constant[FEATURE_LIST]).fit()

    future_dates = pd.date_range(
        start=df_with_quarter_features.index[-1] + pd.offsets.QuarterEnd(),
        periods=num_quarters_to_forecast,
        freq="QE",
    )
    future_df = pd.DataFrame(index=future_dates)
    future_df_with_quarter_features = create_quarter_features(
        future_df, len(historical_data)
    )
    future_df_with_constant = sm.add_constant(
        future_df_with_quarter_features, has_constant="add"
    )
    future_predictions = model.get_prediction(future_df_with_constant[FEATURE_LIST])
    prediction_summary = future_predictions.summary_frame(alpha=1 - confidence_interval)
    if is_financial_data_positive:
        prediction_summary = np.exp(prediction_summary)

    return [
        PredictionIntervalDataPoint(
            date=date.strftime("%Y-%m-%d"),
            financial_data_likeliest_value=row["mean"],
            financial_data_lower_bound=row["obs_ci_lower"],
            financial_data_upper_bound=row["obs_ci_upper"],
        )
        for date, row in prediction_summary.iterrows()
    ]


def _create_dataframe_indexed_by_quarter(historical_data):
    df = pd.DataFrame([data_point.model_dump() for data_point in historical_data])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df
