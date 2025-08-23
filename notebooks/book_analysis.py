# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# %%
# %matplotlib inline
load_dotenv()
current_path = Path.cwd()

# %% [markdown]
# # Load pdf

# %%
pdf_path = current_path / ".." / "data" / "financial_planning.pdf"


loader = PyPDFLoader(pdf_path)
pages = []
async for page in loader.alazy_load():
    pages.append(page)


# %%
page_number = 35
print(f"{pages[page_number].metadata}\n")

# %%
print(pages[page_number].page_content)

# %% [markdown]
# # Vector search

# %%
vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())

# %%
docs = vector_store.similarity_search("P&L", k=2)
for doc in docs:
    print(f"Page {doc.metadata['page']}: {doc.page_content}\n")


# %%
