import os
import logging
import sys
import shutil
import argparse


from llama_index import (
    ObsidianReader,ServiceContext
)


from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import SentenceSplitter

from llama_index.logger import LlamaLogger
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType


import pandas as pd
import time

from langchain.document_loaders import TextLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter


# ****************  Load local var and utils
from config import *

# ****************  MD
from typing import Any, Dict, List, Optional, Tuple, cast
from llama_index.readers.base import BaseReader
import re
from pathlib import Path
from llama_index.schema import Document



#from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
#text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
text_splitter= SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)

node_parser = SimpleNodeParser(
    text_splitter=text_splitter,
    #metadata_extractor=metadata_extractor,
)

service_context = ServiceContext.from_defaults(
    chunk_size=CHUNK_SIZE,
    node_parser=node_parser,
    #llama_logger=llama_logger,
    #callback_manager=callback_manager
    )

DOC_DIRECTORY = "../RessourcesDummy"

from llama_index import  SimpleDirectoryReader
filename_fn = lambda filename: {'file_name': filename}
documents = SimpleDirectoryReader(input_dir=DOC_DIRECTORY, recursive=True, filename_as_id=True, file_metadata=filename_fn).load_data()

import pprint as pp

#from llama_index import Document

if False:
    for d in documents[:4]:
        print("---------------------–\n") 
        
        print("CONTENT\n") 
        pp.pp(d.get_content())
        print("META\n") 
        pp.pp(d.get_metadata_str())
        
        #pp.pp(d.json)
        print("---------------------–\n") 



nodes = node_parser.get_nodes_from_documents(documents)
for n in nodes:
    n.get_content

print(f"Loaded {len(documents)} documents from {DOC_DIRECTORY}")
print(f"Split into {len(nodes)} chunks of text (chunk size {CHUNK_SIZE} - overlap {OVERLAP}). Chunk distribution is as follow: ")

textspd = pd.Series(map(lambda x:len(x.get_content()), nodes ))
stats = textspd.describe()
print(stats)

for n in nodes[:4]:
    print("---------------------–\n") 
    
    print("CONTENT\n") 
    pp.pp(n.get_content())
    print("META\n") 
    pp.pp(n.get_metadata_str())
    
    #pp.pp(d.json)
    print("---------------------–\n") 


# -------------------- 

if False:
    text_splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    docdirectory = DOC_DIRECTORY + "/"

    loader = DirectoryLoader(
        docdirectory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)

    documents = loader.load()
    texts = text_splitter.split_documents(documents)

    print(f"Loaded {len(documents)} documents from {DOC_DIRECTORY}")
    print(f"Split into {len(texts)} chunks of text (max. {CHUNK_SIZE} char each). Chunk distribution is as follow: ")
    textspd = pd.Series(map(lambda x:len(x.page_content), texts))
    stats = textspd.describe()
    print(stats)
