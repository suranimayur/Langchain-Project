# Import required libraries
from dotenv import load_dotenv  # For loading environment variables
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage  # For message types in LangChain
from langchain_core.tools import tool  # For defining tools
from langchain_ollama import ChatOllama  # For using Ollama's LLM
from langsmith import traceable  # For tracing agent execution

# Load environment variables (e.g., API keys)
load_dotenv()

# Constants
MAX_ITERATIONS = 10  # Maximum number of iterations for the agent loop
MODEL = 'gemma4:e4b'  # Model name for Ollama

# --- Tools Section ---
# Tools are functions that the agent can call to perform specific tasks.
# Each tool is decorated with `@tool` to make it callable by the agent.

@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in a predefined catalog.
    Args:
        product (str): Name of the product.
    Returns:
        float: Price of the product, or 0.0 if not found.
    """
    print(f"  >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "mouse": 25.99, "keyboard": 75.99}
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Args:
        price (float): Original price.
        discount_tier (str): Discount tier (bronze, silver, gold).
    Returns:
        float: Final price after discount.
    """
    print(f"  >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discounts_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discounts_percentages.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount / 100), 2)

# --- Agent Loop Section ---
# This function runs the agent loop, which processes the user's question,
# calls tools as needed, and returns the final answer.

@traceable(name="Langchain Agent Loop")  # Enable tracing for this function
def run_agent(question: str):
    """
    Run the agent loop to answer a question using tools.
    Args:
        question (str): User's question.
    Returns:
        str: Final answer or None if max iterations reached.
    """
    # Define the tools available to the agent
    tools = [get_product_price, apply_discount]
    # Create a dictionary for easy tool lookup
    tool_dict = {t.name: t for t in tools}

    # Initialize the LLM (Ollama) and bind the tools to it
    llm = ChatOllama(model=MODEL, temperature=0.5)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # Initialize the conversation with a system message and the user's question
    messages = [
        SystemMessage(
            content=(
                "You are a helpful Shopping assistant."
                "You have access to product catalog tool and discount tool."
                "STRICT RULE: You MUST follow these rules:\n\n"
                "1. Never guess or assume any product price."
                "2. Do not answer questions without using the tools."
                "3. You must call get_product_price tool first to get the product price.\n"
                "4. Only call apply_discount after you have received a price from get_product_price tool."
                "5. If you have price and discount tier, call apply_discount tool."
                "6. If you don't have price or discount tier, call get_product_price tool."
                "7. Do not pass made-up numbers to get_product_price tool."
            )
        ),
        HumanMessage(content=question),
    ]

    # Agent loop: Iterate up to MAX_ITERATIONS times
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"Number of Iterations: {iteration}")
        print(f"{'=' * 60}")

        # Get the LLM's response (which may include tool calls)
        ai_messages = llm_with_tools.invoke(messages)
        print(ai_messages)

        # Extract tool calls from the LLM's response
        tool_calls = ai_messages.tool_calls
        print(tool_calls)

        # If no tool calls, the LLM has provided a final answer
        if not tool_calls:
            print(f"\nThe Final Answer: {ai_messages.content}")
            return ai_messages.content

        # Process the first tool call (for simplicity, we handle one tool per iteration)
        tool_call = tool_calls[0]
        tool_name = tool_call['name']  # Name of the tool to call
        tool_args = tool_call['args']  # Arguments for the tool
        tool_call_id = tool_call['id']  # Unique ID for the tool call

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # Look up the tool by name and call it with the provided arguments
        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # Call the tool and get the result
        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        # Add the LLM's response and the tool's result to the conversation history
        messages.append(ai_messages)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    # If we reach here, the max iterations were exceeded without a final answer
    print("ERROR: Maximum iterations reached without a final answer")
    return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Hello, Langchain Agent (.bind_tools)!")
    print()
    # Example question: Calculate the price of a laptop after applying a gold discount
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(f"Final Price: {result}")