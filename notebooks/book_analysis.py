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
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# %%
# %matplotlib inline
load_dotenv()
current_path = Path.cwd()

# %%
pdf_path = current_path / ".." / "data" / "financial_planning.pdf"

# %% [markdown]
# # Pypdf

# %% [markdown]
# ## PDF Loading (pages)

# %%
loader = PyPDFLoader(pdf_path)
pages = []
async for page in loader.alazy_load():
    pages.append(page)


# %% [markdown]
# ## Basic exploration

# %%
page_number = 35
print(f"{pages[page_number].metadata}\n")

# %%
print(pages[page_number].page_content)

# %% [markdown]
# ## Vector search

# %%
vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())

# %%
docs = vector_store.similarity_search("P&L definition", k=2)
for doc in docs:
    print(f"Page {doc.metadata['page']}: {doc.page_content}\n")


# %% [markdown]
# # PyMuPDF + langchain with simple page concatenation

# %% [markdown]
# ## PDF Loading with blocks concatenation

# %%
doc = fitz.open(pdf_path)
full_text = ""

for page in doc:
    blocks = page.get_text("blocks")
    for block in blocks:
        text_content = block[4]
        full_text += text_content + "\n"

doc.close()

# %%
start_of_pnl_definition_string = "aka Proﬁt"
start_of_pnl_definition_string in full_text

# %% [markdown]
# ## Paragraph extraction

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""] # How to split the text
)

chunks = text_splitter.split_text(full_text)

# %%
for chunk in chunks:
    if start_of_pnl_definition_string in chunk:
        print(chunk)
        print("-------------")

# %% [markdown]
# # PyMuPDF + langchain removing pages headers and footers

# %% [markdown]
# ## Analysis of header and footer size

# %%
doc = fitz.open(pdf_path)

pages = []
for page in doc:
    pages.append(page)

# %%
page = pages[80]
page_height = page.rect.height
print("Page height")
print(page_height)

# %%
blocks = page.get_text("blocks")

# %%
print("First block content")
first_block = blocks[0]
print(first_block[4])
print("First block bottom")
print(first_block[3])

print("------------")

print("Second block content")
second_block = blocks[1]
print(second_block[4])
print("Second block top")
print(second_block[1])

# %%
print("Last block content")
last_block = blocks[-1]
print(last_block[4])
print("Last block top from end of page")
print(page_height - last_block[1])

print("------------")

print("Second last block content")
second_last_block = blocks[-2]
print(second_last_block[4])
print("Second last block bottom from end of page")
print(page_height - second_last_block[3])

# %% [markdown]
# ## PDF loading with removal of header and footer

# %%
# empirical values from previous exploration
header_margin = 60
footer_margin = 60

# %%
doc = fitz.open(pdf_path)
clean_text = ""

for page in doc:
    page_height = page.rect.height
    
    content_y_start = header_margin
    content_y_end = page_height - footer_margin

    blocks = page.get_text("blocks")
    
    for block in blocks:
        # block[1] is the top y-coordinate (y0) of the block
        # block[3] is the bottom y-coordinate (y1) of the block
        
        # Check if the block is fully within the content area
        if block[1] > content_y_start and block[3] < content_y_end:
            clean_text += block[4] # The 5th element is the text content

# %%
start_of_pnl_definition_string in clean_text

# %% [markdown]
# ## Paragraphs extraction

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(clean_text)

# %%
for chunk in chunks:
    if start_of_pnl_definition_string in chunk:
        print(chunk)
        print("-------------")

# %% [markdown]
# ## Vector search

# %%
documents = [Document(page_content=chunk) for chunk in chunks]

vector_store = InMemoryVectorStore.from_documents(documents, OpenAIEmbeddings())

# %%
docs = vector_store.similarity_search("P&L definition", k=2)
for doc in docs:
    print("-----------")
    print(doc.page_content)
    print("-----------")


# %%
docs = vector_store.similarity_search("What is P&L ?", k=2)
for doc in docs:
    print("-----------")
    print(doc.page_content)
    print("-----------")


# %%
docs = vector_store.similarity_search("What is Profit and Loss ?", k=2)
for doc in docs:
    print("-----------")
    print(doc.page_content)
    print("-----------")


# %%
docs = vector_store.similarity_search("What is EBITDA ?", k=2)
for doc in docs:
    print("-----------")
    print(doc.page_content)
    print("-----------")


# %%
docs = vector_store.similarity_search("Show me the last 8 quarters of OPEX costs, then forecast the next 4 quarters, taking into account that the cost there are reasons to think the cost could significantly increase.", k=2)
for doc in docs:
    print("-----------")
    print(doc.page_content)
    print("-----------")


# %%
