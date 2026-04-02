from dotenv import  load_dotenv
load_dotenv()
import os 
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
@tool
def search(query:str) -> str    :
    """Tool that search over internet
    Args:
        Query: The query to be search for

    Returns: The search result     
    """
    
    print(f"Searching for {query}")
    response = tavily_client.search(query=query)
    return response['results'][0]['content']    

llm = ChatOllama(model="minimax-m2.7:cloud")
tools = [TavilySearch()]

agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from build-ai-agents-with-langchain!")
    results = agent.invoke({"messages":HumanMessage(content='What is weather in Tokyo ??')})
    print(results['messages'][-1].content)


if __name__ == "__main__":
    main()
