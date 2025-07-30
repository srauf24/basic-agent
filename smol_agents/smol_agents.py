from smolagents import CodeAgent, OpenAIModel # Or a similar client for OpenAI-compatible APIs
import os
from dotenv import load_dotenv
load_dotenv()
OPEN_ROUTER_TOKEN = os.environ.get("OPENROUTER_API_KEY")

# ... [your other imports and tool definitions] ...

# --- Model Initialization for OpenRouter ---

# This setup tells the client to use OpenRouter's API endpoint
# and your API key (which it finds from the environment variable).
# The model name string is passed to select the specific model.
MODEL_FOR_OPENROUTER = OpenAIModel(
    model_id="meta-llama/llama-3-8b-instruct",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_TOKEN
)

# --- Agent Initialization ---

# Now, create your agent using this new model configuration
simple_agent = CodeAgent(
    tools=[
    ],
    model=MODEL_FOR_OPENROUTER,
)
print("Running agent with OpenRouter model...")

simple_agent.run("What is 125 + 375?")

print("\nTest complete! If you saw the agent's reasoning and the final answer, your setup is working.")

