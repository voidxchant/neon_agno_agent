"""
run_plan.py
Autonomous “MicroSaaS-Miner” agent.
Runs until it has produced a ranked table of 5–10 concrete micro-SaaS ideas.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from tavily import TavilyClient

# --------------------------------------------------
# 1.  ENVIRONMENT & LLM
# --------------------------------------------------
load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0.6)

# --------------------------------------------------
# 2.  TOOLS
# --------------------------------------------------
tavily = TavilyClient()

tools = [
    Tool(
        name="InternetSearch",
        func=lambda q: tavily.search(q, max_results=5),
        description=(
            "Use this to scan the internet for recent Reddit threads, Indie Hackers posts, "
            "Slack/Discord communities, and feature-request forums."
        ),
    )
]

# --------------------------------------------------
# 3.  SYSTEM PROMPT  (the new one)
# --------------------------------------------------
SYSTEM_PROMPT = """
You are “MicroSaaS-Miner,” an expert research agent whose only job is to hunt down **underserved, single-feature, high-margin SaaS opportunities** that fit the 10 formulas below.

INPUT YOU EXPECT FROM ME  
A vertical or audience I care about, default to “remote-first B2B SaaS founders.”

PROCESS (run in order)

1. Fast Forum Scan  
   Search the last 30 days of Reddit, Indie Hackers, Slack/Discord communities, and the top 50 feature-request threads of the dominant platform used by the vertical.  
   Scoring: +2 if the request is > 6 months old with > 30 upvotes / +1 if mentioned in ≥ 3 independent places.

2. Pain-to-Formula Mapping  
   Match every high-scoring pain to the 10 formulas above. Discard anything that needs > 2 weeks of dev or > $1 000 infra per month.

3. Revenue Reality Check  
   Estimate TAM with the “1 000 true fans × $20 MRR” rule. Keep only pains that can realistically reach $10 k MRR within 12 months.

4. Output Table  
   Produce a table with:  
   - Micro-SaaS Name (placeholder)  
   - One-sentence value prop  
   - Pain source link(s)  
   - Closest existing competitor (if any) and its obvious flaw  
   - Suggested MVP stack (no-code or serverless)  
   - 30-second validation test (landing-page headline + call-to-action)

OUTPUT FORMAT  
Return only the table in Markdown. No intro, no summary.
"""

prompt_template = PromptTemplate.from_template(
    f"{SYSTEM_PROMPT}\n\nAvailable tools: {{tools}}\nTool names: {{tool_names}}\n\n{{agent_scratchpad}}"
)

# --------------------------------------------------
# 4.  AGENT + EXECUTOR
# --------------------------------------------------
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

# --------------------------------------------------
# 5.  STATE MANAGEMENT
# --------------------------------------------------
REPORT_FILE = Path("micro_saas_plan.md")

def load_previous() -> str:
    return REPORT_FILE.read_text(encoding="utf-8") if REPORT_FILE.exists() else ""

def append_chunk(text: str) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("a", encoding="utf-8") as f:
        f.write("\n" + text)

# --------------------------------------------------
# 6.  MAIN LOOP
# --------------------------------------------------
def run_until_complete() -> None:
    previous = load_previous()

    if not previous.strip():
        previous = "# Micro-SaaS Ideas Table\n\n"
        append_chunk(previous)

    scratchpad = previous

    while True:
        print("\n[CONTINUING] Resuming with scratchpad length:", len(scratchpad))

        response: Dict[str, Any] = agent_executor.invoke({"input": scratchpad})
        new_text = response.get("output", "")

        finished = "FINISHED" in new_text.upper() or len(new_text) < 50
        append_chunk(new_text)
        scratchpad += new_text

        if finished:
            print("\n[DONE] Plan appears complete.")
            break

        print("\n[LOOP] Checkpoint saved. Restarting agent...\n")

# --------------------------------------------------
# 7.  FIRE IT UP
# --------------------------------------------------
if __name__ == "__main__":
    run_until_complete()