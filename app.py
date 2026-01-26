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
import shutil
import tempfile
from dotenv import load_dotenv
from utils.quiz_generators import QuizGenerator
from utils.flashcard_generator import FlashcardGenerator
from datetime import datetime
load_dotenv()
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"


if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not found.")
    st.write("Please enter your api key to '.env' file")
    st.code("GOOGLE_API_KEY=your_api_key")
    st.stop()


st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-Powered Study Assistant - Transform your learning experience"
    }
)




if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

##Adding stats
if 'stats' not in st.session_state:
    st.session_state.stats={}

    st.session_state.stats.setdefault('questions_asked',0)
    st.session_state.stats.setdefault('quizzes_generated',0)
    st.session_state.stats.setdefault('quizzes_taken',0)
    st.session_state.stats.setdefault('documents_processed',0)
    st.session_state.stats.setdefault('flashcards_generated',0)
        
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()

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
    if not texts:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    return texts

##Initializing the embedding model used to convert text chunks into vector
def get_embeddings():
    """Initialize and return the embedding model"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",task_type="RETRIEVAL_DOCUMENT"
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
    
    from langchain_community.vectorstores import Chroma

def create_vectorstore(texts):
    embeddings = get_embeddings()

    return Chroma.from_documents(
        documents=texts,
        embedding=embeddings
    )
def create_rag_chain(vectorstore):
    """
    Create a RAG chain for question answering eith citation
    """

    ##Configuring a semantic retriever to fetch the most relevant chunks
    retrieval_k = st.session_state.get('settings', {}).get('retrieval_k', 4)
    temperature = st.session_state.get('settings', {}).get('temperature', 0.2)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k}
    )
    def format_docs_with_sources(docs):
        """Format documents and extract source information"""
        formatted_chunks=[]
        sources=set()

        for idx,doc in enumerate(docs):
            formatted_chunks.append(f"[{idx+1}]{doc.page_content}")

            ##Extracting source name
            source= doc.metadata.get('source','Unknown')
            page=doc.metadata.get('page','Unknown')
            sources.add(f"{source} (Page{page})")
        context="\n\n".join(formatted_chunks)
        sources_list= "\n".join([f"-{s}" for s in sources])
        return f"{context}\n\nAvailable Sources:\n {sources_list}"
    

    ##Creating a template for the model to follow to give the desired output
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful study assistant.
        Use ONLY the context below to answer the question.

        Context:
        {context}

        Question:
        {question}

        Instructions:
        1. Answer the question clearly and accurately
        2. After your answer, cite which reference numbers [1], [2], etc. you used
        3. If you're not sure, say "I don't have enough information"
        Answer:
        """
    )

    ##Initializing the chat-based LLM used to generate answers
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=temperature
    )

    ##Creating the RAG pipeline: retrieve context → format prompt → generate answer
    rag_chain = (
        {"context": retriever | format_docs_with_sources,
         "question": RunnablePassthrough()
        }
        | prompt | llm
    )
    
    return rag_chain
##UI Streamlit interface
st.markdown("<h1 style='text-align: center; color: #667eea;'>🎓 AI Study Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #6b7280;'>Transform your PDFs into interactive learning experiences</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header('Upload Documents :')

    uploaded_files=st.file_uploader("Choose PDF files",
        type=['pdf'],  # Only accept PDF files
        accept_multiple_files=True,  # Allow selecting multiple files
        help="Upload one or more PDF files to analyze"
    )
    if uploaded_files:
        st.info(f" {len(uploaded_files)} file(s) selected")
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.write(f". {file.name}")
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing.... This may take a while..."):
                try:
                    # Step 1: Process all PDFs into text chunks
                    texts=process_uploaded_pdfs(uploaded_files)

                    if not texts:
                        st.error("❌ No readable text found in the PDFs.")
                        st.stop()


                    # Step 2: Create vector database from chunks
                    st.session_state.vectorstore= create_vectorstore(texts)
                    # Step 3: Create RAG chain for question answering
                    st.session_state.rag_chain=create_rag_chain(st.session_state.vectorstore)
                    
                    st.session_state.stats['documents_processed'] = len(uploaded_files)

                    # Show success message
                    st.success(f"Processed {len(texts)} chunks from {len(uploaded_files)} document(s)!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        if st.session_state.vectorstore is not None:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history=[]
                st.rerun() #Refreshes the page
        st.markdown("---")
        st.subheader("📊 Session Stats")

        ##Calculating session duration
        duration=datetime.now()-st.session_state.session_start
        total_sec=int(duration.total_seconds())
        minutes=int(total_sec/60)
        seconds=int(total_sec%60)

        #Displaying stats
        col1,col2=st.columns(2)

        with col1:
            st.metric(
                 "❓ Questions", 
            st.session_state.stats['questions_asked']
            )
            st.metric(
                "📝 Quizzes Generated", 
            st.session_state.stats['quizzes_generated']
            )
            st.metric(
            "✅ Quizzes Taken", 
            st.session_state.stats['quizzes_taken']
            )
        with col2:
            st.metric(
            "🎴 Flashcards", 
            st.session_state.stats['flashcards_generated']
            )
            st.metric(
            "📄 Documents", 
            st.session_state.stats['documents_processed']
            )
            st.metric(
            "⏱️ Study Time", 
            f"{minutes}m {seconds}s"
            )
    # Only show chat if documents have been processed
if st.session_state.vectorstore is not None:
    st.markdown("---")  # Horizontal line separator

    with st.expander("Settings⚙️"):
        st.subheader("Retrieval Seettings")

        retrieval_k=st.slider(
            "Number of document chunks to use",
            min_value=2,
            max_value=8,
            value=4,
            help="More cunk s= More context but slower"
        )

        temperature= st.slider(
            "AI Creativity level",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            help="Higher +More creative answers,Lower= More factual"
        )
        st.session_state.settings = {
            'retrieval_k': retrieval_k,
            'temperature': temperature
        }
        # 🔁 Recreate RAG chain with new settings
        st.session_state.rag_chain = create_rag_chain(
            st.session_state.vectorstore
        )
        st.info("Changes apply to new quizzes/flashcards")
    tab1,tab2,tab3= st.tabs(["💬 Chat", "📝 Quiz", "🎴 Flashcards"])

    with tab1:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    ## Chat input box
        if question:=st.chat_input("Ask a question about your documents..."):
            st.session_state.chat_history.append({
                "role":"user",
                "content": question
            })

            st.session_state.stats['questions_asked'] +=1

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..... "):
                    try:
                        response=st.session_state.rag_chain.invoke(question)
                        answer=response.content
                        st.markdown(answer)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                    except Exception as e:
                        error_msg=f"❌ Error: {str(e)}"
                        st.error(error_msg)

    with tab2:
        st.subheader("Generate quiz practice")

        cols1,cols2=st.columns([3,1])
        with cols1:
            quiz_topic=st.text_input(
                "Enter topic for quiz :",
                placeholder="Eg., Photosynthesis,Chapter 4, Artificial Intelligence"
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
                            st.session_state.stats['quizzes_generated'] +=1
                            st.success(f"✅ Generated {len(question)} questions!")
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
                    st.session_state.quiz_answers[idx]= answer
                st.markdown("---")
            ### Submitting button
            if st.button("✅ Submit Quiz", key="submit_quiz"):
                if len(st.session_state.quiz_answers)<len(st.session_state.current_quiz):
                    st.warning("Please answer all questions before submitting!")
                else:
                    st.session_state.stats['quizzes_taken'] +=1
                    correct=0
                    total= len(st.session_state.current_quiz)

                    st.markdown("## 📊 Quiz Results")
                    for idx,q in enumerate(st.session_state.current_quiz):
                        user_answer=st.session_state.quiz_answers.get(idx)
                        correct_answer =q['correct_answer']

                        if user_answer == correct_answer:
                            correct+=1
                            st.success(f"**Q{idx+1}:** ✅ Correct!")
                        else:
                            st.error(f"**Q{idx+1}** ❌ Wrong. Correct answer: {correct_answer}")
                        with st.expander(f"Explanation for Q{idx+1}"):
                            st.write(q['explanation'])
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
            
                    if score_percentage >= 0:
                        st.markdown("---")
                        
                    # Create downloadable quiz report
                    
                        
                    st.session_state.quiz_report = f"""
                    AI STUDY ASSISTANT - QUIZ RESULTS
                    {'='*50}

                    Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    Topic: {quiz_topic}
                    Questions: {total}
                    Correct Answers: {correct}
                    Score: {score_percentage:.1f}%

                    {'='*50}
                    DETAILED RESULTS
                    {'='*50}
                    """

                        
                    # Add each question and result
                    for idx, q in enumerate(st.session_state.current_quiz):
                        user_answer = st.session_state.quiz_answers.get(idx, 'Not answered')
                        correct_answer = q['correct_answer']
                        is_correct = "✓ CORRECT" if user_answer == correct_answer else "✗ WRONG"
                            
                        st.session_state.quiz_report += f"""
                        Question {idx + 1}: {is_correct}
                        {'-'*50}
                        Q: {q['question']}

                        Options:
                        {chr(10).join(q['options'])}

                        Your Answer: {user_answer}
                        Correct Answer: {correct_answer}

                        Explanation:
                        {q['explanation']}

                        {'='*50}
                        """
    
    # Download button

        if "quiz_report" in st.session_state:
            safe_topic = quiz_topic.replace(" ","_")

            st.download_button(
                label="Download Quiz results",
                data=st.session_state.quiz_report,
                file_name=f"quiz_{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                help="Downlaod your quiz results as a text file"
            )
    with tab3:
        st.subheader("🎴 Generate Flashcards")      
        col1,col2=st.columns([3,1])

        with col1:
            flashcard_topic= st.text_input(
                "Enter topic for flashcards:",
                placeholder="eg. Key concepts from Chapter 2",
                key="fc_topic"
            )
        with col2:
            num_cards=st.selectbox(
                "Number of cards:",
                options=[5,10,15,20],
                index=1,key="fc_num"
            )
        if st.button("🎴 Generate Flashcards", type="primary"):
            if not flashcard_topic.strip():
                st.warning("Please enter a topic")
                st.stop()

            else:
                with st.spinner(f"Creating {num_cards} flashcards about {flashcard_topic}..."):
                    try:
                        ##Generate flashcards
                        fc_gen= FlashcardGenerator(st.session_state.vectorstore)
                        flashcards=fc_gen.generate(flashcard_topic,num_cards)

                        if not flashcards:
                            st.error("No flashcards could be generated. Try a different topic.")
                        else:
                            st.session_state.flashcards= flashcards
                            st.session_state.current_card_index=0
                            st.session_state.known_cards= set()
                            st.session_state.stats['flashcards_generated'] += len(flashcards)

                            st.success(f"✅ Created {len(flashcards)} flashcards!")
                    except Exception as e:
                        st.error(f"Error : {str(e)}")

        if 'flashcards' in st.session_state and st.session_state.flashcards:
            st.markdown("---")

            flashcards=st.session_state.flashcards
            current_idx=st.session_state.current_card_index

            st.progress((current_idx+1)/ max(len(flashcards),1))
            st.caption(f"Card {current_idx+1} of {len(flashcards)}")
            ##Initialize flip state if not exists
            if 'card_flipped' not in st.session_state:
                st.session_state.card_flipped=False

            ##Display the flashcards
            card=flashcards[current_idx]
            ##Flashcard container for styling
            st.markdown("""
            <style>
            .flashcard{
                background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                padding: 40px;
                border-radius :15px;
                text-align: center;
                min-height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.2em;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            </style>
            """,unsafe_allow_html=True)

            #Show the front or back based on the flip state
            if not st.session_state.card_flipped:
                st.markdown(f'<div class="flashcard"><b>Q: {card["front"]}</b></div>',unsafe_allow_html=True)
                if st.button("🔄 Flip Card", key="flip"):
                    st.session_state.card_flipped=True
                    st.rerun()
            else:
                st.markdown(f'<div class="flashcard"><b>A: {card["back"]}</b></div>',unsafe_allow_html=True)
                if st.button("🔄 Flip Back", key="flip_back"):
                    st.session_state.card_flipped=False
                    st.rerun()
            st.markdown("") ## For spacing

            col1,col2,col3= st.columns([1,1,1])

            with col1:
                if st.button("⬅️ Previous", disabled=(current_idx == 0)):
                    st.session_state.current_card_index-=1
                    st.session_state.card_flipped=False
                    st.rerun()
            
            with col2:
                if current_idx in st.session_state.known_cards:
                    if st.button("Mark as Unknown"):
                        st.session_state.known_cards.remove(current_idx)
                        st.rerun()
                else:
                    if st.button("Mark as Known"):
                        st.session_state.known_cards.add(current_idx)
                        st.rerun()
            with col3:
                if st.button("➡️ Next", disabled=(current_idx == len(flashcards) - 1)):
                    st.session_state.current_card_index+=1
                    st.session_state.card_flipped= False
                    st.rerun()
                st.markdown("---")
                known_count = len(st.session_state.known_cards)
                total_count= len(flashcards)
                st.metric(
                    "Progress",
                    f"{known_count}/{total_count} cards mastered",
                    f"{(known_count/total_count)*100:.0f}%"
                )

                ##Reset button
                if st.button("Start over again!!"):
                    st.session_state.current_card_index =0
                    st.session_state.known_cards= set()
                    st.session_state.card_flipped=False
                    st.rerun()


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
st.markdown("<p style='text-align: center; color: #6b7280;'>💡 Tip: Ask specific questions for better answer!</p>", unsafe_allow_html=True)