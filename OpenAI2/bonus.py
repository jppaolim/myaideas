 node_postprocessors = [
       PrevNextNodePostprocessor (docstore=docstore, num_nodes=1, mode="both"),
       SimilarityPostprocessor(similarity_cutoff=0.8)
    ]

    retriever = VectorIndexRetriever (
        index=index_vec,
        vector_store_query_mode="mmr", 
        vector_store_kwargs={"mmr_threshold": 0.9},
        similarity_top_k=6,
        node_postprocessors=node_postprocessors
    )   

    from llama_index.response_synthesizers import get_response_synthesizer

    response_synthesizer = get_response_synthesizer(
        response_mode="accumulate",
        #use_async=False,
        text_qa_template=TEXT_QA_TEMPLATE,
        refine_template=CHAT_REFINE_PROMPT,
        service_context=service_context,
        #response_kwargs={'num_children': 2},
     
    )

    query_engine = RetrieverQueryEngine.from_args(   
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        #use_async=False,
        verbose=True
    )




    HYDE_TMPL = (
            "Please write a passage related to the the topic.\n"
            "Keep it short, concise, clear but informative with examples. Now the topic:\n"
            "\n"
            "\n"
            "{context_str}\n"
            "\n"
            "\n"
            'Passage:"""\n'
    )

    from llama_index.prompts.prompt_type import PromptType
    HYDE_PROMPT_TEMPLATE = Prompt(HYDE_TMPL, prompt_type=PromptType.SUMMARY)

    hyde = HyDEQueryTransform(include_original=True, hyde_prompt=HYDE_PROMPT_TEMPLATE)
    hyde_query_engine = TransformQueryEngine(query_engine, hyde)
    response = hyde_query_engine.query(thequery)
    
    print(response)
    print(response.get_formatted_sources())
