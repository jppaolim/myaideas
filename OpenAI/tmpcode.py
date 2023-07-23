CHAT_REFINE_PROMPT_TMPL_MSGS = [

        HumanMessagePromptTemplate.from_template(
        "Here is a topic I am studying: \n"
        "---------------------\n"
        "{query_str}"
        "\n---------------------\n"
        "I have prepared a draft list of bullet points information related to the topic:\n"
        "{existing_answer}\n"
        "I have receveived following new document. Using bullet points summarize its key points that are relevant to the topic and update the draft list accordingly.\n"
        "Bullet points should be crisp and specific but each bullet point should be understandable without context. Try to keep key examples. Here is the new document:\n"
        "---------------------\n"
        "{context_msg}"
        "\n---------------------\n"
        ),
        ]
       
    test=(  "Bullet points should be crisp, short, specific but each bullet point should be understandable without context. Here is the new document:"
      
    )
    
     
    CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
    CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

    # Text QA templates
    DEFAULT_TEXT_QA_PROMPT_TMPL_MSGS =  [
        HumanMessagePromptTemplate.from_template(
        "Here is a topic I am studying: \n"
        "---------------------\n"
        "{query_str}"
        "\n---------------------\n"
        "Using bullet points summarize the key points of following text that are relevant to the topic. Bullet points should be crisp and specific but each bullet point should be understandable without context. Try to keep key examples. \n"
        "Answer with the only bullet points relevant to the topic, no any other consideration.\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        ),
        ]