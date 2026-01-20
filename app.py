##import all the required libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from pathlib import Path
import tempfile
from dotenv import load_dotenv
from utils.quiz_generators import QuizGenerator
load_dotenv()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

##Loading pdfs
def process_uploaded_pdfs(uploaded_files):
    """
    Process multiple uploaded PDF files from Streamlit file uploader
    
    Args:
        uploaded_files: List of uploaded file objects from st.file_uploader()
    
    Returns:
        list: Text chunks from all PDFs combined
    """
    all_documents=[]
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path=tmp_file.name
        loader=PyPDFLoader(tmp_path)
        documents=loader.load()
        for doc in documents:
            doc.metadata['source'] = uploaded_file.name
        all_documents.extend(documents)
        Path(tmp_path).unlink()

##Splitting texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(all_documents)
    return texts

##Initializing the embedding model used to convert text chunks into vector
def get_embeddings():
    """Initialize and return the embedding model"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )


##Storing or loading embeddings into vectorbase using ChromaDB
def create_vectorstore(texts):
    """
    Create a Chroma vectorstore from text chunks
    
    Args:
        texts: List of text chunks from documents
    
    Returns:
        Chroma: Vectorstore with embeddings
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectorstore
def create_rag_chain(vectorstore):
    """
    Create a RAG chain for question answering
    
    Args:
        vectorstore: Chroma vectorstore with document embeddings
    
    Returns:
        Chain: RAG chain for question answering
    """

    ##Configuring a semantic retriever to fetch the most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    ##Creating a template for the model to follow to give the desired output
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful study assistant.
        Use ONLY the context below to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """.strip()
    )

    ##Initializing the chat-based LLM used to generate answers
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0.2
    )

    ##Creating the RAG pipeline: retrieve context → format prompt → generate answer
    rag_chain = (
        {"context": retriever |(lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()
        }
        | prompt | llm
    )
    
    return rag_chain
##UI Streamlit interface
st.title("📚 AI-Powered Study Assistant")
st.markdown("Upload your study materials and ask the questions")
with st.sidebar:
    st.header('Upload Documents :')

    uploaded_files=st.file_uploader("Choose PDF files",
        type=['pdf'],  # Only accept PDF files
        accept_multiple_files=True,  # Allow selecting multiple files
        help="Upload one or more PDF files to analyze"
    )
    if uploaded_files:
        st.info(f".{len(uploaded_files)} file(s) selected")
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.write(f".{file.name}")
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing.... This may take a while..."):
                try:
                    # Step 1: Process all PDFs into text chunks
                    texts=process_uploaded_pdfs(uploaded_files)
                    # Step 2: Create vector database from chunks
                    st.session_state.vectorstore= create_vectorstore(texts)
                    # Step 3: Create RAG chain for question answering
                    st.session_state.rag_chain=create_rag_chain(st.session_state.vectorstore)

                      # Show success message
                    st.success(f"Processed{len(texts)} chunks from {len(uploaded_files)} document(s)!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        if st.session_state.vectorstore is not None:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history=[]
                st.rerun() #Refreshes the page
    # Only show chat if documents have been processed
if st.session_state.vectorstore is not None:
    st.markdown("---")  # Horizontal line separator
    tab1,tab2,tab3= st.tabs(["💬 Chat", "📝 Quiz", "🎴 Flashcards"])

    with tab1:
        for message in st.session_state.chat_history:
            with st.chat_message([message["role"]]):
                st.markdown(message["content"])
    
    ## Chat input box
        if question:=st.chat_input("Ask a question about your documents..."):
            st.session_state.chat_histoy.append({
                "role":"user",
                "content": question
            })

            with st.chat_message("User"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..... "):
                    try:
                        response=st.session_state.rag_chain.invoke(question)
                        answer=response.content
                        st.markdown(answer)

                        st.session_state.chat_history.append({
                            "role": "Assistant",
                            "content": answer
                        })
                    except Exception as e:
                        error_msg=f"❌ Error: {str(e)}"
                        st.error(error_msg)

    with tab2:
        st.subheader("Generate quiz practice")
        from utils.quiz_generators import QuizGenerator

        cols1,cols2=st.columns([3,1])
        with cols1:
            quiz_topic=st.text_input(
                "Enter topic for quiz :",
                "Eg., Photosynthesis,Chapter 4, Artificial Intelligence"
            )
        with cols2:
            num_questions= st.selectbox(
                "Questions:",
                options=[3,5,7,10],index=1
            )
        if st.button("🎲 Generate Quiz", type="primary"):
            if not quiz_topic:
                st.warning("Please enter a topic first!!")
            else:
                with st.spinner(f"Generating {num_questions} questions about {quiz_topic}.."):
                    try:
                        quiz_gen=QuizGenerator(st.session_state.vectorstore)
                        question=quiz_gen.generate(quiz_topic, num_questions)

                        if not question:
                            st.error("No questions could be generated. Try a different topic..")
                        else:
                            st.session_state.current_quiz=question
                            st.session_state.quiz_answers={}
                            st.success(f"✅ Generated {len(questions)} questions!")
                    except Exception as e:
                        st.error(f"Error in generating quiz: {str(e)}")
        if "current_quiz" in st.session_state and st.session_state.current_quiz:
            st.markdown("---")
            for idx,q in enumerate(st.session_state.current_quiz):
                st.markdown(f"### Question {idx+1}")
                st.write(q['question'])

                answer=st.radio(f"Select Your answer:",
                                options=q['options'],
                                key=f"q_{idx}",
                                index=None
                                )
                if answer:
                    st.session_state.quiz_answers[idx]= answer[0]
                st.markdown("---")
            ### Submitting button
            if st.button("✅ Generated {len(questions)} questions!"):
                if len(st.session_state.quiz_answers)<len(st.session_state.current_quiz):
                    st.warning("Please answer all questions before submitting!")
                else:
                    correct=0
                    total= len(st.session_state.current_quiz)

                    st.markdown("## 📊 Quiz Results")
                    for idx,q in enumerate(st.session_state.current_quiz):
                        user_answer=st.session_state.quiz_answers.get(idx)
                        correct_answer =q['correct answer']

                        if user_answer == correct_answer:
                            correct+=1
                            st.success(f"**Q{idx+1}:** ✅ Correct!")
                        else:
                            st.error(f"**Q{idx+1}** ❌ Wrong. Correct answer: {correct_answer}")
                        with st.expander(f"Explaination for Q{idx+1}"):
                            st.write(q['explaination'])
                    score_percentage =(correct/total)*100
                    st.markdown("---")
                    st.markdown(f"### Final Score: {correct}/{total} ({score_percentage:.1f}%)")

                    if score_percentage>=80:
                        st.balloons()
                        st.success("Excellent work! You've got proper understanding of topic")
                    elif score_percentage >=60:
                        st.info("👍 Good job! Review the explanations to improve.")
                    else:
                        st.warning("📚 Keep studying! Review your materials and try again.")
    with tab3:
        st.info("🎴 Flashcard feature coming soon...")                    

    # Display all previous messages from chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):  # "user" or "assistant"
            st.markdown(message["content"])
    
    # Chat input box at bottom of page
    # := is "walrus operator" - assigns AND checks in one line
    if question := st.chat_input("Ask a question about your documents..."):
        
        # Add user's question to chat history
        st.session_state.chat_history.append({"role": "user",
            "content": question})
        
        # Display user's message immediately
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use RAG chain to get answer
                    response = st.session_state.rag_chain.invoke(question)
                    answer = response.content
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Save answer to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg})

else:
    #WELCOME SCREEN: Before any documents uploaded
    st.info("👈 **Get Started:** Upload PDF documents in the sidebar to begin!")
    
    # Show instructions
    st.markdown("""
    ### How to use:
    1. 📤 Upload one or more PDF files using the sidebar
    2. 🚀 Click "Process Documents" to analyze them
    3. 💬 Ask questions about your study materials
    4. 🎯 Get instant answers with context from your documents
    
    ### Features:
    - 📚 Multi-document support
    - 🔍 Semantic search across all documents
    - 💡 Context-aware answers
    - 📝 Chat history
    """)
    
    # Show example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the main topic of chapter 3?
        - Summarize the key concepts in section 2
        - What are the important formulas mentioned?
        - Explain the theory of [topic] from my notes
        """)

# Footer
st.markdown("---")
st.caption("💡 Tip: Ask specific questions for better answers!")