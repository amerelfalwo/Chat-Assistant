from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from app.services.vectorstore import get_vectorstore
from app.services.memory_manager import get_session_history
from app.core.config import settings

def get_conversational_rag():
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.0
    )
    
    vectorstore = get_vectorstore()
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
    You are **MediBot**, an AI-powered assistant trained to help users understand medical documents and health-related questions.
    
    Your job is to provide clear, accurate, and helpful responses based **only on the provided context**.
    
    ---
    Context:
    {context}
    ---
    Instructions:
    - Respond in a calm, factual, and respectful tone.
    - Use simple explanations when needed.
    - If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
    - Do NOT make up facts.
    - Do NOT give medical advice or diagnoses.
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
