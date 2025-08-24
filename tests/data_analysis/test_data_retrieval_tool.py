from data_analysis.data_retrieval_tool import read_data_by_topic, FinancialDataTopics


def test_read_data_by_topic_should_be_able_to_read_revenue_data():
    # Given
    topic = FinancialDataTopics.REVENUE

    segments = [
        "A1",
        "Other",
        "Editions",
        "Marketplaces",
        "Small+",
        "Local",
        "Mid Market",
        "Small",
        "Enterprise",
    ]
    expected_dimensions = ["date", "Total Revenue"] + segments

    # When
    result = read_data_by_topic.invoke({"topic": topic})

    # Then
    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, dict) for item in result)
    assert all(dimension in result[0] for dimension in expected_dimensions)
