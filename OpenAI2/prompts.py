from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
) 
from llama_index.prompts import Prompt


chat_text_qa_msgs_Bullet = [
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


chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs_Bullet)
TEXT_QA_TEMPLATE_BULLET = Prompt.from_langchain_prompt(chat_text_qa_msgs_lc)


chat_text_qa_msgs_Writer = [
            SystemMessagePromptTemplate.from_template(
                "You are an assistant to an important executive decision maker. You write blog posts for him. They are sharp, crisp but engageing and illustrated with original analogies and examples. They use short sentences and sometimes bullet points to be fluid and engageing."
            ),
            HumanMessagePromptTemplate.from_template(
            "The goal is to prepare a post on the following topic:\n"
            "\n"
            "{query_str}"
            "\n"
            "Use the specific and relevant knowledge, insights or examples of below document to prepare the best post you can. If nothing is relevant within document, just write NOTHING RELEVANT and stop there. Here is the document:\n"
            "\n"
            "{context_str}"
            "\n"
            ),
            ]


chat_text_qa_msgs_lc2 = ChatPromptTemplate.from_messages(chat_text_qa_msgs_Writer)
TEXT_QA_TEMPLATE_BLOG = Prompt.from_langchain_prompt(chat_text_qa_msgs_lc2)

#------------- NOT USED 

chat_text_qa_msgs_Insight = [
    SystemMessagePromptTemplate.from_template(
        "You are an assistant to an important executive decision maker. You present relevant information to him so he makes better thinking."
    ),
    HumanMessagePromptTemplate.from_template(
    "The goal is to get a point of view on the following topic:\n"
    "\n"
    "{query_str}"
    "\n"
    "Capture the key information, insights or examples included in the below document which are relevant to the topic. If nothing is relevant within document, just answer with NOT RELEVANT and stop there.\n" 
    "Make your answer concise and sharp. Make it specific by selecting interesting or original examples from the document. Use short sentences to be fluid and engageing. In any case don't add knowledge, only what you gather from the document.\n"
    "Here is the document:\n"
    "\n"
    "{context_str}"
    "\n"
    ),
    ]

chat_text_qa_msgs_lc3 = ChatPromptTemplate.from_messages(chat_text_qa_msgs_Insight )
TEXT_QA_TEMPLATE_INSIGHT = Prompt.from_langchain_prompt(chat_text_qa_msgs_lc3)

