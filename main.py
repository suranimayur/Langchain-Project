from dotenv import  load_dotenv
load_dotenv()
import os 
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

@tool
def search_query(query:str):
    """Tool thast search for a Internet
    
    Args: 
        The query to search for

    Returns:
        The search result 
    
    """
    print(f'Searching for {query}')
    return f"Tokyo weather is sunny"

llm=  ChatOllama()

tools = [search_query]

agent = create_agent(model= llm,tools = tools)

if __name__ == "__main__":
    main()
