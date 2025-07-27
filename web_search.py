from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    VisitWebpageTool,
    OpenAIModel,
)
import os
from dotenv import load_dotenv
load_dotenv()
OPEN_ROUTER_TOKEN = os.environ.get("OPENROUTER_API_KEY")


MODEL_FOR_OPENROUTER = OpenAIModel(
    model_id="tngtech/deepseek-r1t2-chimera:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_TOKEN

)

search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=MODEL_FOR_OPENROUTER,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=MODEL_FOR_OPENROUTER,
    managed_agents=[search_agent],
)
manager_agent.run(
    "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"
)