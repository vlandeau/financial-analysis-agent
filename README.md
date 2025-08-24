# Financial Data Analysis Agent

## Setup

### Package Installation

We are using `uv` to manage our virtual environments and package dependencies. Please refer to the [official documentation](https://docs.astral.sh/uv/) for more information on how to install and use `uv`.
Please run the folloing command to install the project : 

```bash
uv run pip install -e .
```

### Environment Variables

We are using `python-dotenv` to manage our environment variables. 

In order to setup your environment variables, you should copy the `.env.template` file to a new file named `.env` and fill in your API keys.

### Input data

Please copy the `P&L.xlsx` and `Financial Planning & Analysis and Performance Management ( PDFDrive ).pdf` file to `data/`, and rename `Financial Planning & Analysis and Performance Management ( PDFDrive ).pdf`to `financial_planning.pdf`.

## Exploration and data processing

Notebooks have been used both to explore the data, possible information extraction from the pdf and forecasting models, 
but also to process data from the excel file and store it to parquet files to be used by the agent. 

### Running Jupyter Notebooks

We are using Jupyter notebooks for exploration and experimentation. You can find the notebooks in the `notebooks` directory. 

In order to run jupyter, you can use the following command:

```bash
uv run jupyter notebook
```

### Version Control

We are also using [jupytext](https://jupytext.readthedocs.io/en/latest/) to manage our Jupyter notebooks as text files. This allows for easier tracking and review of the notebooks' code evolution. We only track the `.py` files in git, while the `.ipynb` files are generated from them.

This means the first time you retrieve the repository, you won't see any `.ipynb` file. In order to generate the `.ipynb` files from the `.py` files, you can either:

1. Run jupyter, then right click on some `.py` file and click on `Open with` > `Notebook`.

2. Use the following command to generate all `.ipynb` files at once:

```bash
uv run jupytext --to ipynb notebooks/*.py
```

## Running the Agent
You can run the agent using the following command:

```bash
uv run analysis --user-query "<your query here>"
```

The agent has two tools at its disposal:
* read_data_by_topic: a tool to retrieve the relevant data, based on the parquet files generated from the excel file.
* forecast_future_financial_data: a tool to forecast future financial data based on historical data.

The forecasting tool generate a confidence interval for the forecasted values, which is then used by the agent to provide a more informative response, with optimistic and pessimistic scenarios.
The LLM can configure the confidence level to use, based on the user query.
