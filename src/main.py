import click
import os
from data_analysis.data_retrieval_tool import read_data_by_topic
from forecasting.forecasting_tool import forecast_future_financial_data
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import cast, Any


load_dotenv()


@click.command()
@click.option(
    "--user-query",
    prompt="Enter your financial query",
    help="The financial query to be processed by the agent.",
)
def main(user_query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [read_data_by_topic, forecast_future_financial_data]
    system_prompt = """
    You are a financial analyst agent. Your task is to assist users in analyzing financial data and making forecasts based on historical trends.
    Please use the provided tools to retrieve data and perform forecasting as needed.
    When using the tools, please keep in mind that the data is stored according to the last date of each quarter (e.g., Q1 ends on March 31, Q2 ends on June 30, etc.).

    If the user request a realistic forecasting or does not specify the type of forecasting, you should only give them the likeliest value of the prediction interval.
    If the user request an optimistic forecasting, you should only give them the upper bound of the prediction interval.
    If the user request a pessimistic forecasting, you should only give them the lower bound of the prediction interval.
    You should also adjust the confidence interval based on the user's perception of the economic situation:
    - If the user does not specify any change in the economic situation, use a confidence interval of 95%.
    - If the user indicates that the economic situation could significantly improve or worsen, adjust the confidence interval to 98% or more.
    Do not indicate how the predictions were computed if the user does not explicitly ask for it.
    
    Most but not all financial data is are stored in total and by customer segment. The customer segments are:
    * Enterprise
    * A1
    * Editions
    * Marketplaces
    * Local
    * Mid Market
    * Small+
    * Small
    * Other
    """.strip()

    debug = os.getenv("DEBUG", "false").strip().lower() == "true"
    financial_analyst_agent = create_react_agent(
        llm, tools, prompt=system_prompt, debug=debug
    )
    input_data = {
        "messages": [
            HumanMessage(content=user_query)
        ],
    }
    agent_response = financial_analyst_agent.invoke(
        cast(Any, input_data) # used to avoid IDE warnings
    )
    agent_answer = agent_response["messages"][-1].content
    print(agent_answer)
    return agent_answer


if __name__ == "__main__":
    main()
