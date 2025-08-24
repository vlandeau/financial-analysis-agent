from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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


@tool(args_schema=TrainAndForecastInput)
def forecast_future_financial_data(historical_data: List[TimeSeriesDataPoint], num_quarters_to_forecast: int) -> List[TimeSeriesDataPoint]:
    """
    Trains a simple forecasting model on historical financial data and predicts future values.

    Args:
        historical_data (List[TimeSeriesDataPoint]): A list of historical data points, each with a date and financial data value.
        num_quarters_to_forecast (int): The number of future quarters to forecast after training on the historical data.

    Returns:
        List[TimeSeriesDataPoint]: A list of forecasted data points for the specified number of future quarters.
    """
    df = pd.DataFrame([data_point.model_dump() for data_point in historical_data])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    model = ExponentialSmoothing(df['financial_data_value'], seasonal='add', seasonal_periods=4)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=num_quarters_to_forecast)
    
    return [
        TimeSeriesDataPoint(date=str(date.date()), financial_data_value=value)
        for date, value in forecast.items()
    ]
