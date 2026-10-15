from dotenv import load_dotenv
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json

from typing import Literal, Optional, List, Dict, Any, Type, Annotated
from langgraph.graph import MessagesState
import os
from prompts import plan_prompt, executor_prompt, agent_system_prompt
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL


load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class State(MessagesState):
  user_query: Optional[str]
  enabled_agents: Optional[List[str]]
  plan: Optional[List[Dict[int, Dict[str, Any]]]]
  current_step: int
  agent_query: Optional[str]
  last_reason: Optional[str]
  replan_flag: Optional[bool]
  replan_attempts: Optional[Dict[int, Dict[int, int]]]


reasoning_llm = ChatOpenAI(
    model="o3",
    model_kwargs={"response_format": {"type": "json_object"}}
)


def planner_node(state: State) -> Command[Literal['executor']]:
  """
  Runs the planning LLM and stores the resulting plan in the state
  """
  llm_reply = reasoning_llm.invoke([plan_prompt(state)])

  try:
    content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
    parsed_plan = json.loads(content_str)
  except json.JSONDecodeError:
    raise ValueError(
        f"Planner returned an invalid JSON: \n{llm_reply.content}"
    )

  replan = state.get("replan_flag", False)
  updated_plan: Dict[str, Any] = parsed_plan

  return Command(
      update={
          "plan": updated_plan,
          "messages": [HumanMessage(content=llm_reply.content, name="replan" if replan else "initial_plan")],
          "user_query": state.get("user_query", state["messages"][0].content), "current_step": 1 if not replan else state["current_step"],
          "replan_flag": state.get("replan_flag", False),
          "last_Reason": "",
          "enabled_agents": state.get("enabled_agents")

      },
      goto="executor"
  )


#executor node
MAX_REPLANS = 5

def executor_node(state: State) -> Command[Literal['web_researcher', 'chart_generator', 'synthesizer', 'planner']]:

  plan: Dict[str, Any] = state.get("plan", {})
  step: int = state.get("current_step", 1)

  if state.get("replan_flag"):
    planned_agent = plan.get(str(step), {}).get("agent")
    return Command(
        update={
            "replan_flag": False,
            "current_step": step + 1,
        },
        goto=planned_agent
    )
  llm_reply = reasoning_llm.invoke([executor_prompt(state)])
  try:
    content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
    parsed = json.loads(content_str)
    replan:bool = parsed["replan"]
    goto:str = parsed["goto"]
    reason:str = parsed["reason"]
    query:str = parsed["query"]
  except Exception as e:
    raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from e

  updates: Dict[str, Any] = {
      "messages": [HumanMessage(content=llm_reply.content, name="executor")],
      "last_reason": reason,
      "agent_query": query,
  }

  #replan accounting
  replans: Dict[int, int] = state.get("reply_attempts", {}) or {}
  step_replans  = replans.get(step, 0)

  #replan decision
  if replan:
    if step_replans < MAX_REPLANS:
      replans[step] = step_replans + 1
      updates.update({
          "replan_attempts": replans,
          "replan_flag": True,
          "current_step": step,
      })
      return Command(update=updates, goto="planner")
    else:
      next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
      updates["current_step"] = step + 1
      return Command(update = updates, goto=next_agent)
  planned_agent = plan.get(str(step), {}).get("agent")
  updates["current_step"] = step + 1 if goto == planned_agent else step
  updates["replan_flag"] = False
  return Command(update=updates, goto=goto)


#tavily tool defination for the web search agent
tavily_tool = TavilySearch(max_results=5)

llm = ChatOpenAI(model="gpt-4o")

#research agent

web_search_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=agent_system_prompt(f"""
        You are the Researcher. You can ONLY perform research
        by using the provided search tool (tavily_tool).
        When you have found the necessary information, end your output.
        Do NOT attempt to take further actions.
    """),
)

#web search agent node

def web_research_node(
    state: State,
) -> Command[Literal["executor"]]:
  agent_query = state.get("agent_query")
  result  = web_search_agent.invoke({"messages": agent_query})
  goto="executor"

  result["messages"][-1] = HumanMessage(
      content = result["messages"][-1].content, name="web_researcher"
  )

  return Command(
      update={
          "messages": result["messages"]
      },
      goto=goto
  )

repl = PythonREPL()

@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart"],):
  """
  A Python shell. Use this to execute python code.
  """
  try:
    result = repl.run(code)
  except BaseException as e:
    return f"Failed to execute. Error: {repr(e)}"
  result_str = (
        f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    )
  return (
        result_str
        + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=agent_system_prompt(
        """
        You can only generate charts. You are working with a researcher
        colleague.
        1) Print the chart first.
        2) Save the chart to a file in the current working directory.
        3) At the very end of your message, output EXACTLY two lines
        so the summarizer can find them:
           CHART_PATH: <relative_path_to_chart_file>
           CHART_NOTES: <one concise sentence summarizing the main insight in the chart>
        Do not include any other trailing text after these two lines.
        """
    )
)

def chart_node(state: State) -> Command[Literal["chart_summarizer"]]:
  result = chart_agent.invoke(state)

  result["messages"][-1] = HumanMessage(
      content=result["messages"][-1].content, name="chart_generator"
  )
  goto="chart_summarizer"
  return Command(
      update={
          "messages": result["messages"],
      },
      goto=goto
  )

chart_summary_agent = create_react_agent(
    llm,
    tools=[],
    prompt=agent_system_prompt(
        "You can only generate image captions. You are working with a researcher colleague and a chart generator colleague. "
        + "Your task is to generate a standalone, concise summary for the provided chart image saved at a local PATH, where the PATH should be and only be provided by your chart generator colleague. The summary should be no more than 3 sentences and should not mention the chart itself."
    ),
)

def chart_summary_node(
    state: State,
) -> Command[Literal[END]]:
  result = chart_summary_agent.invoke(state)
  print(f"Chart summarizer answer: {result['messages'][-1].content}")
  goto = END
  return Command(
      update={
          "messages": result["messages"],
          "final_answer": result["messages"][-1].content,
      },
      goto=goto
  )


def synthesizer_node(state: State) -> Command[Literal[END]]:
    """
    Creates a concise, human‑readable summary of the entire interaction,
    **purely in prose**.

    It ignores structured tables or chart IDs and instead rewrites the
    relevant agent messages (research results, chart commentary, etc.)
    into a short final answer.
    """
    # Gather informative messages for final synthesis
    relevant_msgs = [
        m.content for m in state.get("messages", [])
        if getattr(m, "name", None) in ("web_researcher",
                                        "chart_generator",
                                        "chart_summarizer")
    ]

    user_question = state.get("user_query", state.get("messages", [{}])[0].content if state.get("messages") else "")

    synthesis_instructions = (
        """
        You are the Synthesizer. Use the context below to directly
        answer the user's question. Perform any lightweight calculations,
        comparisons, or inferences required. Do not invent facts not
        supported by the context. If data is missing, say what's missing
        and, if helpful, offer a clearly labeled best-effort estimate
        with assumptions.\n\n
        Produce a concise response that fully answers the question, with
        the following guidance:\n
        - Start with the direct answer (one short paragraph or a tight bullet list).\n
        - Include key figures from any 'Results:' tables (e.g., totals, top items).\n
        - If any message contains citations, include them as a brief 'Citations: [...]' line.\n
        - Keep the output crisp; avoid meta commentary or tool instructions.
        """
        )

    summary_prompt = [
        HumanMessage(content=(
            f"User question: {user_question}\n\n"
            f"{synthesis_instructions}\n\n"
            f"Context:\n\n" + "\n\n---\n\n".join(relevant_msgs)
        ))
    ]

    llm_reply = llm.invoke(summary_prompt)

    answer = llm_reply.content.strip()
    print(f"Synthesizer answer: {answer}")

    return Command(
        update={
            "final_answer": answer,
            "messages": [HumanMessage(content=answer, name="synthesizer")],
        },
        goto=END,           # hand off to the END node
    )

from langgraph.graph import START, StateGraph

workflow = StateGraph(State)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("web_researcher", web_research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("chart_summarizer", chart_summary_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "planner")

graph = workflow.compile()


import json

query = "Chart the current market capitalization of the top 5 banks in the US?"
print(f"Query: {query}")

state = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "enabled_agents": ["web_researcher", "chart_generator",
                               "chart_summarizer", "synthesizer"],
        }
graph.invoke(state)

print("--------------------------------")

query = "Identify current regulatory changes for the financial services industry in the US."
print(f"Query: {query}")

state = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "enabled_agents": ["web_researcher", "chart_generator",
                               "chart_summarizer", "synthesizer"],
        }
graph.invoke(state)

print("--------------------------------")
