"""
Flashcard Generator for AI Study Assistant
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
class FlashcardGenerator:
    def __init__(self,vectorstore):
        self.vectorstore= vectorstore

        self.llm=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.6
        )
        self.template="""Based on the following context,create {num_cards} flashcards.
Context:
{context}
Important : Return Only a Valid Json array with no markdown formatting.
Format exactly like this:
[
    {{
    "front":"What is photosynthesis?",
    "back":"Photosynthesis is the process by which plants convert light energy into chemical energy."
    }}
]
Generate {num_cards} flashcards now:"""
        self.prompt= ChatPromptTemplate.from_template(self.template)
    def generate (self,topic,num_cards=10):
        try:
            docs=self.vectorstore.similarity_search(topic, k=3)

            if not docs:
                return[]
            context="\n\n".join([doc.page_content for doc in docs])

            chain= self.prompt| self.llm
            response= chain.invoke({
                "context" : context,
                "num_cards": num_cards
            })
            content= response.content.strip()
            if content.startswith("```"):
                content=content.split("```")[1]
                if content.startswith("json"):
                    content=content[4:].strip()
            flashcards=json.loads(content)
            return flashcards
        except Exception as e:
            print(f"Error: {e}")
            return []
