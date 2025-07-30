#!/usr/bin/env python3
"""
duck_duck_go.py  – one‑file demo

Requirements (install once):
    pip install smolagents ddgs beautifulsoup4 requests python-dotenv

.env file must contain:
    OPENROUTER_API_KEY=<your key>

What the script does:
  • Defines two Tool subclasses (WebSearchTool, WebScrapeTool)
  • Registers them with a smol‑agents CodeAgent
  • Lets an LLM (via OpenRouter) call the tools to answer a question
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. imports  (keep __future__ first)                                         #
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS                      # ← new package name

from smolagents import CodeAgent, OpenAIModel, Tool

# ─────────────────────────────────────────────────────────────────────────────
# 2. env & logging                                                            #
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPEN_ROUTER_TOKEN = os.getenv("OPENROUTER_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s › %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Tool subclasses                                                          #
# ─────────────────────────────────────────────────────────────────────────────
class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Run a DuckDuckGo search and return up to `max_results` hits as a list "
        "of strings, each formatted '<title> – <url>'."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query, e.g. 'best mosques in California'.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 10).",
            "nullable": True,          # optional
        },
    }
    output_type = "object"             # list[str] is allowed via 'object'

    def forward(self, query: str, max_results: int = 10) -> List[str]:  # type: ignore[override]
        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=max_results)
        return [f"{h['title']} – {h['href']}" for h in hits]


class WebScrapeTool(Tool):
    name = "web_scrape"
    description = (
        "Download `url` and return the visible text. "
        "If a CSS `selector` is provided, only that element's text is returned. "
        "Returns an empty string on HTTP errors."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "Web page to fetch.",
        },
        "selector": {
            "type": "string",
            "description": "Optional CSS selector for a specific element.",
            "nullable": True,          # optional
        },
    }
    output_type = "string"

    def forward(self, url: str, selector: Optional[str] = None) -> str:  # type: ignore[override]
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("web_scrape: %s", e)
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        if selector:
            node = soup.select_one(selector)
            return node.get_text(" ", strip=True) if node else ""

        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. model & agent setup                                                      #
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FOR_OPENROUTER = OpenAIModel(
    model_id="meta-llama/llama-3-8b-instruct",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_TOKEN,
)

agent = CodeAgent(
    tools=[WebSearchTool(), WebScrapeTool()],
    model=MODEL_FOR_OPENROUTER,
    additional_authorized_imports=["json", "re"],  # allow regex/json in sandbox
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. quick test                                                               #
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    question = (
        "Search for the best mosque recommendations for a trip to California."
    )
    answer = agent.run(question)
    print("\nAssistant:\n", answer)
