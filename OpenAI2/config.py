import os
import logging

from dotenv import load_dotenv
load_dotenv()


## SRC DOC
#DOC_DIRECTORY = "../RessourcesDummy"
DOC_DIRECTORY = "../Ressources"

## STORAGE
PERSIST_DIRECTORY="./db"

##EMBEDDINGS
INSTRUCT_MODEL="hkunlp/instructor-large"
#needed for HF 
TOKENIZERS_PARALLELISM=False

CHUNK_SIZE=768 ## pas clair ce qu'il faut mettre et on a des token vs des char ac 512 on a en gros 1680 characters 
MAXTOKEN = 4096
OVERLAP = 256 

## llm 
BASE_MODELPATH="../models/"
#Base one : good 
#LOCALMODEL=BASE_MODELPATH + "wizardlm-13b-v1.1.ggmlv3.q4_0.bin"
#MLOCALODEL=BASE_MODELPATH + "wizardlm-30b.ggmlv3.q4_0.bin"

#Llama 2  : good 
#LOCALMODEL=BASE_MODELPATH +  "nous-hermes-llama2-13b.ggmlv3.q4_0.bin"
LOCALMODEL=BASE_MODELPATH +  "redmond-puffin-13b.ggmlv3.q4_0.bin"

#open AI
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
MODEL="gpt-3.5-turbo"


## PROMTPS
BASE_PROMPTPATH="./Prompts/"
PROMPTFILEQA=BASE_PROMPTPATH + "Base.txt"
PROMPTFILEREFINE=BASE_PROMPTPATH + "Refine.txt"

## Behavior 
LOGLEVEL=logging.DEBUG

#rebuild database 
FORCE_REBUILD= False

#test with fake LLM 
FAKELLM=False