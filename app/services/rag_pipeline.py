from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from app.services.vectorstore import get_vectorstore
from app.services.memory_manager import get_session_history
from app.core.config import settings

def get_conversational_rag(session_id: str):
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    vectorstore = get_vectorstore(namespace=session_id)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Advanced Retrieval: Multi-Query
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    
    # History-Aware Reformulation Setup
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, advanced_retriever, contextualize_q_prompt
    )
    
    # Main QA Prompt
    qa_system_prompt = """
    You are an **Expert Research & Document Assistant**.
    
    Your job is to deeply analyze the provided document context and accurately perform tasks such as answering questions or generating comprehensive articles.
    
    ---
    Context:
    {context}
    ---
    Instructions:
    - Base your responses **only** on the provided context above. 
    - If the user asks a direct question, provide a precise and accurate answer.
    - If the user asks you to write an article or a summary, act as an "Expert Writer". Generate a well-structured article using Markdown headings (e.g., #, ##), bullet points, and highly coherent paragraphs based upon the context.
    - If the context does not contain the information required to safely or fully address the prompt, you MUST clearly apologize and state logically what information is missing from the uploaded documents. Do NOT hallucinate.
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Wrap with memory
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain
