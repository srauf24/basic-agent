import os
import json
from smolagents import (
    CodeAgent,
    OpenAIModel,
    Tool,
    tool,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
)
from smolagents.models import ChatMessage

from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """You are Alfred, the meticulous and resourceful butler of Wayne Manor. Your primary goal is to assist with party planning by leveraging the tools at your disposal.

Think step-by-step and break down the problem into smaller, manageable actions. Always consider your next action carefully.

You have access to the following tools. You must call these tools by generating a JSON blob with an "action" key (the tool name) and an "action_input" key (a dictionary of arguments for the tool).

Available Tools:
- **web_search**:
  Description: Performs a web search using DuckDuckGo. Useful for finding general information, URLs, or specific data on the internet.
  Arguments: {"query": {"type": "string", "description": "The search query."}}
  Example Usage:
  ```json
  {
    "action": "web_search",
    "action_input": {"query": "best villain masquerade party Spotify playlist"}
  }
  ```

- **suggest_menu**:
  Description: Suggests a menu based on the occasion.
  Arguments: {"occasion": {"type": "string", "description": "The type of party (e.g., 'casual', 'formal', 'superhero', 'custom')."}}
  Example Usage:
  ```json
  {
    "action": "suggest_menu",
    "action_input": {"occasion": "formal"}
  }
  ```

- **catering_service_tool**:
  Description: Returns the highest-rated catering service in Gotham City.
  Arguments: {"query": {"type": "string", "description": "A search term for finding catering services."}}
  Example Usage:
  ```json
  {
    "action": "catering_service_tool",
    "action_input": {"query": "Gotham City best catering"}
  }
  ```

- **superhero_party_theme_generator**:
  Description: Suggests creative superhero-themed party ideas based on a category.
  Arguments: {"category": {"type": "string", "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic gotham')."}}
  Example Usage:
  ```json
  {
    "action": "superhero_party_theme_generator",
    "action_input": {"category": "villain masquerade"}
  }
  ```

Your interaction will follow this strict Thought-Action-Observation cycle:

Question: The input question you need to answer.
Thought: You should always think step-by-step about what to do next. Break down the problem. Consider which tool to use, if any, and what arguments it needs.
Action:
```json
$JSON_BLOB_FOR_TOOL_CALL
```
Observation: The result of the action. This Observation is unique, complete, and the source of truth.
(This Thought/Action/Observation cycle can repeat N times. You should take several steps when needed. The $JSON_BLOB_FOR_TOOL_CALL must be formatted as markdown and only use a SINGLE action at a time.)

Once you have gathered all necessary information and are confident in your answer, you must end your output with the following format:

Thought: I now know the final answer.
Final Answer: The complete and concise answer to the original input question, incorporating all relevant information from your observations.

Now begin!
"""

## ------------------ ##
## Custom Tool Definitions
## ------------------ ##

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of party (e.g., 'casual', 'formal').
    """
    occasion = occasion.lower()
    if "casual" in occasion:
        return "Pizza, snacks, and drinks."
    elif "formal" in occasion:
        return "3-course dinner with wine and dessert."
    elif "superhero" in occasion:
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.
    Args:
        query (str): A search term for finding catering services.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    best_service = max(services, key=services.get)
    return f"The best catering service is {best_service} with a rating of {services[best_service]}."

#json 
class SuperheroPartyThemeTool(Tool):
    """A tool to suggest superhero-themed party ideas."""
    name = "superhero_party_theme_generator"
    description = "Suggests creative superhero-themed party ideas based on a category."
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade').",
        }
    }
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests dress as DC heroes with themed cocktails.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as Batman villains.",
            "futuristic gotham": "Neo-Gotham Night: A cyberpunk-style party with neon decorations."
        }
        return themes.get(category.lower(), "Theme not found. Try 'classic heroes' or 'villain masquerade'.")

## ------------------ ##
## Main Execution
## ------------------ ##

if __name__ == "__main__":
    # 1. Get API Key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Please set it in your environment or a .env file.")

    # 2. Initialize the Model for OpenRouter
    print(" Initializing model for OpenRouter...")

    LLAMA_MODEL = OpenAIModel(
        model_id="meta-llama/llama-3-8b-instruct",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        )

    # 3. Assemble all the tools for the agent
    all_tools = [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool(),
    ]

    # We will create a dictionary to map tool names (as strings) to their actual Python callable objects.
    tool_map = {
        "suggest_menu": suggest_menu,
        "catering_service_tool": catering_service_tool,
        "superhero_party_theme_generator": SuperheroPartyThemeTool().forward, # Custom Tool's method
    }

    # 4. Initialize the "Alfred" Agent (CodeAgent will still be used, but its behavior is guided by the prompt)
    # The CodeAgent will now be responsible for parsing the JSON and executing the corresponding tool.
    alfred_agent = CodeAgent(
        tools=all_tools, # all_tools still contains DuckDuckGoSearchTool()
        model=LLAMA_MODEL,
    )

    # 5. Define the task and run the agent
    prompt_text = (
        "Give me the best playlist for a party at Wayne Manor. "
        "The party's theme is a 'villain masquerade'."
    )
    print(f"\n▶️  Running agent with prompt: '{prompt_text}'")
    print("-" * 60)

    # --- Agent Execution Loop (Conceptual) ---
    # This part will be more complex, involving multiple calls to the LLM
    # and parsing its responses.

    current_messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompt_text),
    ]

    while True:
        # First, get the LLM's Thought and Action
        llm_response = alfred_agent.model.generate(messages=current_messages) # Simplified call
        response_content = llm_response.content # Assuming this structure

        print(f"LLM Response:\n{response_content}")
        current_messages.append(ChatMessage(role="assistant", content=response_content))

        # Check if the LLM is providing a Final Answer
        if "Final Answer:" in response_content:
            break # Exit loop if final answer is given

        # Parse the Action JSON
        try:
            # Extract JSON from markdown block
            json_start = response_content.find("```json")
            json_end = response_content.find("```", json_start + 1)
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start + len("```json"):json_end].strip()
                action_data = json.loads(json_str)
                action_name = action_data.get("action")
                action_input = action_data.get("action_input", {})

                if action_name and action_name in tool_map:
                    tool_function = tool_map[action_name]
                    print(f"Executing tool: {action_name} with input: {action_input}")
                    observation_result = tool_function(**action_input) # Execute the tool
                    print(f"Observation: {observation_result}")

                    # Append Observation to messages for next LLM call
                    current_messages.append(ChatMessage(role="tool_response", content=observation_result)) # Or "user" role depending on model
                else:
                    print(f"Error: Tool '{action_name}' not found or invalid action format.")
                    current_messages.append(ChatMessage(role="tool_response", content="Error: Invalid tool call or tool not found."))
            else:
                print("Error: No valid JSON action found in LLM response.")
                current_messages.append(ChatMessage(role="tool_response", content="Error: No valid JSON action found."))

        except json.JSONDecodeError:
            print("Error: Invalid JSON in LLM response.")
            current_messages.append(ChatMessage(role="tool_response", content="Error: Invalid JSON format."))
        except Exception as e:
            print(f"An error occurred during tool execution: {e}")
            current_messages.append(ChatMessage(role="tool_response", content=f"Error during tool execution: {e}"))