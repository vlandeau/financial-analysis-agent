from enum import Enum


class FinancialDataTopics(str, Enum):
    COST_OF_SERVICES = "cost_of_services"
    EBITDA = "ebitda"
    OPEX = "opex"
    TRANSACTIONS = "transactions"
    CREDIT_CARD_FEES = "credit_card_fees"
    GROSS_PROFIT = "gross_profit"
    REVENUE = "revenue"


TOPIC_TO_FILENAME_MAP = {
    FinancialDataTopics.COST_OF_SERVICES: "cost_of_services.parquet",
    FinancialDataTopics.EBITDA: "ebitda.parquet",
    FinancialDataTopics.OPEX: "opex.parquet",
    FinancialDataTopics.TRANSACTIONS: "transactions.parquet",
    FinancialDataTopics.CREDIT_CARD_FEES: "credit_card_fees.parquet",
    FinancialDataTopics.GROSS_PROFIT: "gross_profit.parquet",
    FinancialDataTopics.REVENUE: "revenue.parquet",
}