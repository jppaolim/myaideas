#### fichier qui fonctionne mais on extrait trop de meta data, pas forcément nécessaire, et je dois refaire toute la BDD
#### par ailleurs OpenAI marche bcp mieux à la fin quand même 

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
    ObsidianReader,
    ServiceContext, 
    StorageContext, 
    load_index_from_storage,
    Prompt,
    #ResponseSynthesizer,
    PromptHelper
)

from langchain.llms import  LlamaCpp

from langchain.llms.fake import FakeListLLM 

from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import SentenceSplitter

#from llama_index.response_synthesizers import get_response_synthesizer


from langchain.chat_models import ChatOpenAI

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from llama_index.logger import LlamaLogger
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType



# ****************  Load local var and utils
from config import *


#from trulens_eval import TruLlama
from llama_index.llms import OpenAI

def read_str_prompt(filepath: str):

    with open(filepath, 'r') as file:
            template = file.read()

    return(template) 

from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings

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
    
    llm = LlamaCpp(model_path=LOCALMODEL, n_threads=15,  n_ctx=MAXTOKEN, max_tokens=MAXOUTPUT,  temperature = 0.3, top_k = 50, top_p=0.70, last_n_tokens_size=256,  n_batch=1024, repeat_penalty=1.17, use_mmap=True, use_mlock=True, n_gpu_layers=1)
    llm.client.verbose= False
    llm_predictor = LLMPredictor(llm=llm)


    from llama_index.node_parser.extractors import (
        MetadataExtractor,
        SummaryExtractor,
        TitleExtractor,
        KeywordExtractor,
        MetadataFeatureExtractor,
    )
    
    metadata_extractor = MetadataExtractor(
        extractors=[
            #TitleExtractor(nodes=5),
            SummaryExtractor(summaries=["prev", "self","next"]),
            #KeywordExtractor(keywords=5),
        ],
    )

    #from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
    #text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
    text_splitter= SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)

    node_parser = SimpleNodeParser(
        text_splitter=text_splitter,
        metadata_extractor=metadata_extractor,
    )


    # ***************  Service Context 
    #   
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model= embed_model,
        prompt_helper=PromptHelper(context_window=MAXTOKEN-150,   num_output=MAXOUTPUT, chunk_overlap_ratio=0.2),
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
   
   

    #for d in documents:
    #    d.excluded_llm_metadata_keys = ['file_name']
    #    d.excluded_embed_metadata_keys = ['file_name']


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

    qa_template = Prompt(read_str_prompt(PROMPTFILEQA))
    re_template = Prompt(read_str_prompt(PROMPTFILEREFINE))

    retriever = VectorIndexRetriever (
        index=index_vec,
        vector_store_query_mode="mmr", 
        vector_store_kwargs={"mmr_threshold": 0.9},
        similarity_top_k=6

    )   
     
    #from llama_index.indices.postprocessor import PrevNextNodePostprocessor
 
    #docstore=storage_context.docstore
    #node_postprocessor = PrevNextNodePostprocessor (docstore=docstore, num_nodes=1, mode="both")
 
    from llama_index.response_synthesizers import get_response_synthesizer

    response_synthesizer = get_response_synthesizer(
        response_mode="accumulate",
        use_async=False,
        text_qa_template=qa_template,
        refine_template=re_template,
        service_context=service_context,
        #response_kwargs={'num_children': 2},
        #node_postprocessors=[node_postprocessor],
     
    )

    query_engine = RetrieverQueryEngine.from_args(   
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        use_async=False,
        verbose=True
    )

#    query_engine = index_vec.as_query_engine (
#        response_mode="accumulate",
#        use_async=False,
#        verbose=True,
#        similarity_top_k=3,
#        text_qa_template=qa_template,
#        refine_template=re_template,
#        service_context=service_context,
#        #response_kwargs={'num_children': 2},
#        #node_postprocessors=[node_postprocessor],
#       )

    from llama_index.response.pprint_utils import pprint_source_node
    nodes = retriever.retrieve(thequery)
    for n in nodes:
        pprint_source_node(n, source_length=2500, wrap_width=300)

    #l = TruLlama(query_engine)

    response  = query_engine.query(thequery)
    print(response)
    #print(response.get_formatted_sources())

    #response  = l.query(thequery)
    #print(response)
    #exit()


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str, required=True, help="Query string to process")
    args = parser.parse_args()
    main(args.query)