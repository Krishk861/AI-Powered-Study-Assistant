"""
Quiz Generator for AI Assistant
Generates miltiple choice questions(MCQs) from the material provided
"""
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from langchain_core.prompts import ChatPromptTemplate
import json

class QuizGenerator:
    def __init__(self,vectorstore):
        """
        Initialize quiz generator

        Args:
            vectorstore: Chroma vector database with study materials
        """
        self.vectorstore= vectorstore

        self.llm = ChatNVIDIA(
            model="nvidia/nemotron-3-super-120b-a12b",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.6
        )
        self.template="""
        
        Based on the following context from study materials, generate {num_questions} multiple choice questions.

        Context:
        {context}

        IMPORTANT: Return ONLY a valid JSON array with no markdown formatting, explanation, or backticks.

        Format each question exactly like this:
        [
        {{
            "question": "What is photosynthesis?",
            "options": ["A) Process of...", "B) Method of...", "C) Type of...", "D) Form of..."],
            "correct_answer": "A",
            "explanation": "Brief explanation why this is correct"
        }}
        ]
        Generate {num_questions} questions now:
        """
        self.prompt=ChatPromptTemplate.from_template(self.template)

    def generate(self,topic,num_questions=5):
        """
        Generate quiz questions on a specific topic
        
        Args:
            topic: Subject to generate questions about
            num_questions: Number of questions to generate (default 5)
        
        Returns:
            list: List of question dictionaries, or empty list if error
        """
        try:
            docs=self.vectorstore.similarity_search(topic,k=3)
            if not docs:
                return []
            context="\n\n".join([doc.page_content for doc in docs])

            chain=self.prompt | self.llm
            response =chain.invoke({
                "context" : context,
                "num_questions": num_questions
            })

            content=response.content.strip()
            if content.startswith("```"):

                content=content.split("```")[1]
                if content.startswith("json"):
                    content=content[4:].strip()
            questions=json.loads(content)
            return questions
            
        except Exception as e:
            print("Quiz generation failed:", repr(e))
            raise