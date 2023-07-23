import os
import logging
import sys
import shutil
import argparse


from llama_index import (
    VectorStoreIndex,
    OpenAIEmbedding,
    LangchainEmbedding,
    LLMPredictor, 
    ServiceContext, 
    StorageContext, 
    load_index_from_storage,
    Prompt,
    PromptHelper
)


from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings

from llama_index.node_parser.simple import SimpleNodeParser

from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from llama_index.langchain_helpers.text_splitter import SentenceSplitter

from llama_index.logger import LlamaLogger
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
)

from llama_index.indices.postprocessor import PrevNextNodePostprocessor, SimilarityPostprocessor

from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine


# ****************  Load local var and utils
from config import *

from datetime import datetime

def log_interaction(input_str, response_str, filename):
    # Open the file in append mode ('a')
    with open(filename, 'a') as f:
        # Write the current date and time
        f.write(f"{datetime.now()}:\n")
        # Write the input
        f.write(f"Input: {input_str}\n")
        # Write the response
        f.write(f"Response: {response_str}\n")
        # Write a new line for separating this interaction from the next one
        f.write("-------------------------------\n")
        f.write("\n")

#from trulens_eval import TruLlama

def read_str_prompt(filepath: str):

    with open(filepath, 'r') as file:
            template = file.read()

    return(template) 


def embeddings_function():
    embed_instruction = "Represent the document for retrieval; Input: "
    query_instruction = "Represent the topic or query for retrieving relevant documents; Input: "
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
    Instructembedding = HuggingFaceInstructEmbeddings(
        model_name=INSTRUCT_MODEL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, embed_instruction=embed_instruction, query_instruction=query_instruction)
    return Instructembedding

#************* - A casser dans class MarkdownReader
import re
def correct_markdown(self, content: str) -> str:
        """Correct headers inside markdown links."""
        pattern = r"\[\s*\n*(#{1,6}.*?)\n*\]\((.*?)\)"
        # Split the header and the link, then reformulate
        def replacer(match):
            header = match.group(1)
            link = match.group(2)
            return header + "\n[Link](" + link + ")"
        content = re.sub(pattern, replacer, content)
        return content
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
def parse_tups(
    self, filepath: Path, errors: str = "ignore"
) -> List[Tuple[Optional[str], str]]:
    """Parse file into tuples."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if self._remove_hyperlinks:
        content = self.remove_hyperlinks(content)
    if self._remove_images:
        content = self.remove_images(content)
    content = self. correct_markdown(content)
    markdown_tups = self.markdown_to_tups(content)
    return markdown_tups

# *************** MAIN LOOP 

def main(thequery: str):

    # ***************  logging and Callback 

    logger = logging.getLogger()
    logger.setLevel(LOGLEVEL)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler('ouput.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    llama_logger = LlamaLogger()

    # ***************  Embedding
    #Instructor : 
    embed_model = LangchainEmbedding(embeddings_function())
    
    # ***************  LLM     

    MAXOUTPUT = MAXTOKEN / 2 

    llm=ChatOpenAI(temperature=0.6, model_name=MODEL, streaming=False, max_tokens=MAXOUTPUT) 
    #llm.client.verbose= False
    llm_predictor = LLMPredictor(llm=llm)

   
    metadata_extractor = MetadataExtractor(
        extractors=[
            SummaryExtractor(summaries=["prev", "self","next"]),
        ],
    )

    text_splitter= SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)

    ## on va pas faire les summary 
    node_parser = SimpleNodeParser(
        text_splitter=text_splitter,
        #metadata_extractor=metadata_extractor,
    )

    # ***************  Service Context 
    #   
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model= embed_model,
        prompt_helper=PromptHelper(context_window=MAXTOKEN-150,   num_output=MAXOUTPUT, chunk_overlap_ratio=0.3),
        chunk_size=CHUNK_SIZE,
        node_parser=node_parser,
        llama_logger=llama_logger,
        callback_manager=callback_manager
        )
    
    #from llama_index import set_global_service_context
    #set_global_service_context(service_context) 
        
    # ***************  Load Documents, Build Index 
           
    docfile  = PERSIST_DIRECTORY+"/docstore.json" 
    indexfile = PERSIST_DIRECTORY+"/index_store.json"
    missingfile = not (os.path.exists(docfile) and os.path.exists(indexfile))

    from llama_index import  SimpleDirectoryReader
    filename_fn = lambda filename: {'file_name': filename}

    documents = SimpleDirectoryReader(input_dir=DOC_DIRECTORY, recursive=True, filename_as_id=True, file_metadata=filename_fn).load_data()
   
   
    #if something missing or "force build", we start from scratch 
    if (missingfile or FORCE_REBUILD):
 
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(documents)
 
        index_vec = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
       
        index_vec.set_index_id("Vector")
        index_vec.storage_context.persist(persist_dir=PERSIST_DIRECTORY) 
      
    #if we have something we refresh 
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIRECTORY)
        index_vec = load_index_from_storage(storage_context, index_id="Vector", service_context=service_context) 
        
        index_vec.refresh_ref_docs(documents,
            update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
        )
        index_vec.storage_context.persist(persist_dir=PERSIST_DIRECTORY) 


    # ***************  Make the query with the real good templates 

    from langchain.prompts.chat import (
         ChatPromptTemplate,
         HumanMessagePromptTemplate,
         SystemMessagePromptTemplate,
    )
    
    from llama_index.prompts import Prompt

    #from llama_index.prompts.prompts import RefinePrompt, QuestionAnswerPrompt

    chat_text_qa_msgs = [
        SystemMessagePromptTemplate.from_template(
            "You are an assistant to an important executive decision maker. You prepare summaries based only on bullet points. Each bullet point should be detailed enough to be understandable without context and capture a specific key idea with examples."
        ),
        HumanMessagePromptTemplate.from_template(
        "I prepare a memo on the following topic:\n"
        "\n"
        "{query_str}"
        "\n"
        "Using bullet points capture the key information of following document that are relevant to the topic."
        "Answer only with the bullet points relevant to the topic, no any other consideration. Here is the document:\n"
        "\n"
        "{context_str}"
        "\n"
        ),
        ]
    
    chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    TEXT_QA_TEMPLATE = Prompt.from_langchain_prompt(chat_text_qa_msgs_lc)

 
    docstore=storage_context.docstore
    
    node_postprocessors = [
       PrevNextNodePostprocessor (docstore=docstore, num_nodes=1, mode="both"),
       SimilarityPostprocessor(similarity_cutoff=0.75)
        ]

    query_engine = index_vec.as_query_engine (
        response_mode="accumulate",
        vector_store_query_mode="mmr", 
        vector_store_kwargs={"mmr_threshold": 0.9},
        similarity_top_k=10,
        use_async=False,
        verbose=True,
        node_postprocessors=node_postprocessors,
        text_qa_template=TEXT_QA_TEMPLATE,
        #refine_template=CHAT_REFINE_PROMPT,
        service_context=service_context,
       )

    #from llama_index.response.pprint_utils import pprint_source_node
    #nodes = retriever.retrieve(thequery)
    #for n in nodes:
    #    pprint_source_node(n, source_length=2500, wrap_width=300)

    #l = TruLlama(query_engine)

    response  = query_engine.query(thequery)
    
    print(response)
    print(response.get_formatted_sources())


    filled_template = (
        "Please write a passage related to the the topic.\n"
        "Keep it under 200 words, sharp and clear. Now the topic:\n"
        "\n"
        "\n"
        "{txt}\n"
        "\n"
        "\n"
        'Passage:"""\n'
    )
    
    from langchain import LLMChain

    system_message = "You are an assistant to an important executive decision maker"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
    human_template=filled_template
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    result = chain.run(thequery)

    print(result)

    requery = re.sub('^assistant:\s*', '', result)
    
    response2  = query_engine.query(requery)
    
    print(response2)
    print(response2.get_formatted_sources())

    
    #response  = l.query(thequery)
    #print(response)
    #exit()

    log_interaction(thequery, result , "interactions.txt")
    log_interaction(thequery, response.response_txt() , "interactions.txt")
    log_interaction(thequery, response2.response_txt() , "interactions.txt")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str, required=True, help="Query string to process")
    args = parser.parse_args()
    main(args.query)