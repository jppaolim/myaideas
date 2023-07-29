import os
import logging
import sys
import shutil
import argparse
import re

# ****************  Load local var and utils
from .config import *
from .utils import *
from .prompts import *


# ****************  LLAMA imports 

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
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
)
from llama_index.indices.postprocessor import PrevNextNodePostprocessor, SimilarityPostprocessor

from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

from llama_index.logger import LlamaLogger
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

# ***************  logging and Callback 

logger = logging.getLogger()
logger.setLevel(LOGLEVEL)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
llama_logger = LlamaLogger()

# ***************  Query Engine Builder

def run_query(sim_cut: float, mmr: float, topk: int, text_qa_template: str, index_vec: VectorStoreIndex, service_context: ServiceContext, query: str):
    """
    Builds a query engine with the given parameters and runs the query.
    """
    node_postprocessors = [
       SimilarityPostprocessor(similarity_cutoff=sim_cut)
        ]

    query_engine = index_vec.as_query_engine (
        response_mode="accumulate",
        vector_store_query_mode="mmr", 
        vector_store_kwargs={"mmr_threshold": mmr},
        similarity_top_k=topk,
        use_async=False,
        verbose=True,
        node_postprocessors=node_postprocessors,
        text_qa_template=text_qa_template,
        service_context=service_context,
       )
    return query_engine.query(query)

def main(thequery: str):
    """
    Main loop of the program.
    Includes the embedding, LLM, text splitter & node parser, service context, loading documents, building index, and making the query with the real good templates.
    """
    # Embedding
    embed_instruction = "Represent the document for retrieval; Input: "
    query_instruction = "Represent the topic or query for retrieving relevant documents; Input: "
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
     
    embed_model = LangchainEmbedding(
        HuggingFaceInstructEmbeddings(
        model_name=INSTRUCT_MODEL, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs, 
        embed_instruction=embed_instruction, query_instruction=query_instruction
        )
    )
    
    # LLM     
    MAXOUTPUT = MAXTOKEN / 2 
    llm=ChatOpenAI(temperature=0.6, model_name=MODEL, streaming=False, max_tokens=MAXOUTPUT) 
    llm_predictor = LLMPredictor(llm=llm)

    # Text splitter & node parser  
    text_splitter= SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    metadata_extractor = MetadataExtractor(
        extractors=[
            SummaryExtractor(summaries=["prev", "self","next"]),
        ],
    )

    node_parser = SimpleNodeParser(
        text_splitter=text_splitter,
    )

    # Service Context 
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model= embed_model,
        prompt_helper=PromptHelper(context_window=MAXTOKEN-150,   num_output=MAXOUTPUT, chunk_overlap_ratio=0.3),
        chunk_size=CHUNK_SIZE,
        node_parser=node_parser,
        llama_logger=llama_logger,
        callback_manager=callback_manager
        )
    
    # Load Documents, Build Index 
    docfile  = PERSIST_DIRECTORY+"/docstore.json" 
    indexfile = PERSIST_DIRECTORY+"/index_store.json"
    missingfile = not (os.path.exists(docfile) and os.path.exists(indexfile))

    from llama_index import  SimpleDirectoryReader
    filename_fn = lambda filename: {'file_name': filename}

    documents = SimpleDirectoryReader(input_dir=DOC_DIRECTORY, recursive=True, filename_as_id=True, file_metadata=filename_fn).load_data()
   
    # If something missing or "force build", we start from scratch 
    if (missingfile or FORCE_REBUILD):
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(documents)
 
        index_vec = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
       
        index_vec.set_index_id("Vector")
        index_vec.storage_context.persist(persist_dir=PERSIST_DIRECTORY) 
      
    # If we have something we refresh 
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIRECTORY)
        index_vec = load_index_from_storage(storage_context, index_id="Vector", service_context=service_context) 
        
        index_vec.refresh_ref_docs(documents,
            update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
        )
        index_vec.storage_context.persist(persist_dir=PERSIST_DIRECTORY) 

    # Make the query with the real good templates 
    results = []                                                                                                                                                     
    if RUN_BASEQUERY:
        sim_cut  = 0.85
        mmr = 0.95

        response_A = run_query(
            sim_cut=sim_cut,
            mmr=mmr,
            topk=6,
            text_qa_template=TEXT_QA_TEMPLATE_BULLET,
            index_vec=index_vec,
            service_context=service_context,
            query=thequery
        )
        
        results.append(log_interaction(f"SET A : bullet points - very relevant : {sim_cut}, mmr =  {mmr}", response_A))  

    if RUN_EXPANDED1:
        sim_cut  = 0.50
        mmr = 0.90

        response_B = run_query(
            sim_cut=sim_cut,
            mmr=mmr,
            topk=6,
            text_qa_template=TEXT_QA_TEMPLATE_BULLET,
            index_vec=index_vec,
            service_context=service_context,
            query=thequery
        )
        
        results.append(log_interaction(f"SET B : bullet points - more diverse : {sim_cut}, mmr =  {mmr}", response_B))                                               


    if RUN_EXPANDED2:
        sim_cut  = 0.50
        mmr = 0.90

        response_C = run_query(
            sim_cut=sim_cut,
            mmr=mmr,
            topk=6,
            text_qa_template=TEXT_QA_TEMPLATE_BLOG,
            index_vec=index_vec,
            service_context=service_context,
            query=thequery
        )
        
        results.append(log_interaction(f"SET C : Blog - more diversity : {sim_cut}, mmr =  {mmr}", response_C))                                                      


    if RUN_HYDE:
        sim_cut  = 0.50
        mmr = 0.90

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

        vanilla  = chain.run(thequery)
        log_interaction("VANILLA")  
        log_interaction(thequery, vanilla)
        requery = re.sub('^assistant:\s*', '', vanilla)

        response_D = run_query(
            sim_cut=sim_cut,
            mmr=mmr,
            topk=6,
            text_qa_template=TEXT_QA_TEMPLATE_BULLET,
            index_vec=index_vec,
            service_context=service_context,
            query=requery
        )
        
        results.append(log_interaction(f"SET D - HYDE TENTATIVE INSPIRATION : {sim_cut}, mmr =  {mmr}", response_D))      

    return results