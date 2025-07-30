

import logging
import os
import string
from dotenv import load_dotenv
from smolagents import CodeAgent,  OpenAIModel, WebSearchTool, VisitWebpageTool, DuckDuckGoSearchTool
import tenacity
load_dotenv()
OPEN_ROUTER_TOKEN = os.environ.get("OPENROUTER_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s › %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_FOR_OPENROUTER = OpenAIModel(
    model_id="meta-llama/llama-3-8b-instruct",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_TOKEN

)
simple_agent = CodeAgent(
    tools=[WebSearchTool(),VisitWebpageTool(),
    ],
    model=MODEL_FOR_OPENROUTER,
    additional_authorized_imports=[ 'json', 're'],

)

# Initialize the agent with the correct model and strict instructions
simple_agent.run("Search for the best mosque recommendations for a trip to California.")
