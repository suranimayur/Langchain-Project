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
from pydantic import BaseModel, Field
from typing  import List


class Source(BaseModel):
    """ Schema for source used by agent"""

    url:str = Field(description="The URL of source ")

class Agentresponse(BaseModel):
    """ Schema for agent response with answer and sources list """

    answer:str = Field(description="The agents answer to the query")
    sources:List[Source] = Field(default_factory=list,description="The list of sources used to generate the answer")



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

llm = ChatOllama(model="gemma4:e4b")
tools = [TavilySearch()]

agent = create_agent(model=llm,tools=tools,response_format=Agentresponse)

def main():
    print("Hello from build-ai-agents-with-langchain!")
    results = agent.invoke({"messages":HumanMessage(content='What is weather in Pune now ?? Prepare professional report')})
    print(results['messages'][-1].content)


if __name__ == "__main__":
    main()
