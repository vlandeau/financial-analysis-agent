from data_analysis.data_retrieval_tool import read_data_by_topic
from forecasting.forecasting_tool import forecast_future_financial_data
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [read_data_by_topic, forecast_future_financial_data]
system_prompt = """
You are a financial analyst agent. Your task is to assist users in analyzing financial data and making forecasts based on historical trends.
Please use the provided tools to retrieve data and perform forecasting as needed.
When using the tools, please keep in mind that the data is stored according to the last date of each quarter (e.g., Q1 ends on March 31, Q2 ends on June 30, etc.).
""".strip()


financial_analyst_agent = create_react_agent(
    llm, tools, prompt=system_prompt, debug=True,
)


if __name__ == "__main__":
    agent_response = financial_analyst_agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Show me the last 8 quarters of revenue, then forecast the next 4 quarters.",
        },],
    })
    print(agent_response["messages"][-1].content)
