# Financial Data Analysis Agent

## Setup

### Package Installation

We are using `uv` to manage our virtual environments and package dependencies. Please refer to the [official documentation](https://docs.astral.sh/uv/) for more information on how to install and use `uv`.

### Environment Variables

We are using `python-dotenv` to manage our environment variables. 

In order to setup your environment variables, you should copy the `.env.template` file to a new file named `.env` and fill in your API keys.

## Exploration

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
