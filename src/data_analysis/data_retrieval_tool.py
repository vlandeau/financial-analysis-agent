from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Hashable
import os
from pathlib import Path
import pandas as pd


from data_analysis.financial_data_topics import (
    FinancialDataTopics,
    TOPIC_TO_FILENAME_MAP,
)


DATA_FOLDER = Path(__file__).parent.parent.parent / "data" / "output"


class ReadDataByTopicInput(BaseModel):
    """Input schema for the data retrieval tool."""

    topic: FinancialDataTopics = Field(
        ..., description="The specific, predefined data topic to retrieve."
    )


@tool(args_schema=ReadDataByTopicInput)
def read_data_by_topic(
    topic: FinancialDataTopics,
) -> Optional[List[Dict[Hashable, Any]]]:
    filename = TOPIC_TO_FILENAME_MAP.get(topic)

    if not filename:
        return None

    file_path = os.path.join(DATA_FOLDER, filename)

    df = pd.read_parquet(file_path)
    df.index.name = "date"
    df.reset_index(inplace=True)
    return df.to_dict(orient="records")
