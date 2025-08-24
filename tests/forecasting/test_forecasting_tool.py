from forecasting.forecasting_tool import (
    forecast_future_financial_data,
    TimeSeriesDataPoint,
)


def test_forecast_future_financial_data_with_seasonality_and_no_trend():
    # Given
    historical_data = [
        TimeSeriesDataPoint(date="2020-03-31", financial_data_value=100.0),
        TimeSeriesDataPoint(date="2020-06-30", financial_data_value=110.0),
        TimeSeriesDataPoint(date="2020-09-30", financial_data_value=120.0),
        TimeSeriesDataPoint(date="2020-12-31", financial_data_value=130.0),
        TimeSeriesDataPoint(date="2021-03-31", financial_data_value=100.0),
        TimeSeriesDataPoint(date="2021-06-30", financial_data_value=110.0),
        TimeSeriesDataPoint(date="2021-09-30", financial_data_value=120.0),
        TimeSeriesDataPoint(date="2021-12-31", financial_data_value=130.0),
    ]
    num_quarters_to_forecast = 4

    # When
    forecast = forecast_future_financial_data.invoke(
        {
            "historical_data": historical_data,
            "num_quarters_to_forecast": num_quarters_to_forecast,
        }
    )

    # Then
    assert len(forecast) == num_quarters_to_forecast

    expected_dates = ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]
    for i in range(num_quarters_to_forecast):
        assert forecast[i].date == expected_dates[i]

    expected_values = [100.0, 110.0, 120.0, 130.0]
    for i in range(num_quarters_to_forecast):
        assert (
            abs(forecast[i].financial_data_likeliest_value - expected_values[i]) < 1e-5
        )
        assert abs(forecast[i].financial_data_lower_bound - expected_values[i]) < 1e-5
        assert abs(forecast[i].financial_data_upper_bound - expected_values[i]) < 1e-5
