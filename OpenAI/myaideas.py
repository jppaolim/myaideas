import os
import logging
import sys
import shutil
import argparse


from llama_index import (
    VectorStoreIndex,
    OpenAIEmbedding,
    LLMPredictor, 
    ServiceContext, 
    StorageContext, 
    load_index_from_storage,
    Prompt,
    ResponseSynthesizer,
    PromptHelper
)

from langchain.llms.fake import FakeListLLM 

from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import SentenceSplitter

from langchain.chat_models import ChatOpenAI

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from llama_index.logger import LlamaLogger
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType


# ****************  Load local var and utils
from config import *
from utils import read_str_prompt


    
# *************** MAIN LOOP 

def main(thequery: str):

    # ***************  logging and Callback 

    #logger = logging.getLogger()
    #logger.setLevel(LOGLEVEL)

    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #stdout_handler = logging.StreamHandler(sys.stdout)
    #stdout_handler.setLevel(logging.INFO)
    #stdout_handler.setFormatter(formatter)

    #file_handler = logging.FileHandler('ouput.log')
    #file_handler.setLevel(logging.DEBUG)
    #file_handler.setFormatter(formatter)

    #logger.addHandler(file_handler)
    #logger.addHandler(stdout_handler)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    llama_logger = LlamaLogger()


    # ***************  LLM     
    import openai
    openai.api_key=os.getenv('OPENAI_API_KEY')

    #MODEL="gpt-4"
    if not FAKELLM:
        llm=ChatOpenAI(temperature=0.6, model_name=MODEL, streaming=False) 
      
    else :
        responses=[
        "This is a great summary 1",
        "This is a great summary 2",
        "This is a great summary 3",
        "This is a great summary 4",
        "This is a great summary 5" ,
        "This is a great summary 6",
        "This is a great summary 7",
        "This is a great summary 8"
        ]

        llm = FakeListLLM(responses=responses)

    llm_predictor = LLMPredictor(llm=llm)

    # ***************  Service Context 
    #   
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=PromptHelper(context_window=4097-150,   num_output=MAXTOKEN, chunk_overlap_ratio=0.5),
        chunk_size=CHUNK_SIZE,
        embed_model= OpenAIEmbedding(openai_api_key=OPENAI_API_KEY),
        node_parser=SimpleNodeParser(SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)),
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

    documents = SimpleDirectoryReader(input_dir=DOC_DIRECTORY, recursive=True, filename_as_id=True).load_data()


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

    #qa_template = Prompt(read_str_prompt(PROMPTFILEQA))
    #re_template = Prompt(read_str_prompt(PROMPTFILEREFINE))

    from langchain.prompts.chat import (
        AIMessagePromptTemplate,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
    )

    from llama_index.prompts.prompts import RefinePrompt, QuestionAnswerPrompt

    CHAT_REFINE_PROMPT_TMPL_MSGS = [

        HumanMessagePromptTemplate.from_template(
        "I am preparing a summary on the following topic for a decision maker:\n"
        "\n"
        "{query_str}"
        "\n"
        "Here is my current summary:\n"
        "\n"
        "{existing_answer}\n"
        "\n"
        "I have receveived following new document. Add new bullet points to the summary or update some of them to take into account this document."
        "Eeach bullet point should be relevant to the topic, detailed enough to be understandable without context by the decision maker and capture a specific key idea with examples. Here is the new document:\n"
        "\n"
        "{context_msg}"
        "\n"
        ),
        ]
       
    test=(  "Bullet points should be crisp, short, specific but each bullet point should be understandable without context. Here is the new document:"
      
    )


    CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
    CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

    # Text QA templates
    DEFAULT_TEXT_QA_PROMPT_TMPL_MSGS =  [
        HumanMessagePromptTemplate.from_template(
        "I am preparing a summary on the following topic for a decision maker:\n"
        "\n"
        "{query_str}"
        "\n"
        "Using bullet points capture the key information of following document that are relevant to the topic."
        "Each bullet point should be detailed enough to be understandable without context by the decision maker and capture a specific key idea with examples."
        "Answer only with the bullet points relevant to the topic, no any other consideration. Here is the document:\n"
        "\n"
        "{context_str}"
        "\n"
        ),
        ]
    
    TEXT_QA_TEMPLATE_LC = ChatPromptTemplate.from_messages(DEFAULT_TEXT_QA_PROMPT_TMPL_MSGS)
    TEXT_QA_TEMPLATE = QuestionAnswerPrompt.from_langchain_prompt(TEXT_QA_TEMPLATE_LC)

    from llama_index.indices.postprocessor import PrevNextNodePostprocessor
 
    docstore=storage_context.docstore
    node_postprocessor = PrevNextNodePostprocessor (docstore=docstore, num_nodes=1, mode="both")
 

    query_engine = index_vec.as_query_engine (
        verbose=True,  
        similarity_top_k=18,
        text_qa_template=TEXT_QA_TEMPLATE,
        refine_template=CHAT_REFINE_PROMPT,
        response_mode='tree_summarize',
        response_kwargs={'num_children': 2},
        service_context=service_context,
        #node_postprocessors=[node_postprocessor],
        #streaming=True
    
    )

    response  = query_engine.query(thequery)
    print(response)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str, required=True, help="Query string to process")
    args = parser.parse_args()
    main(args.query)