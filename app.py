import logging
from typing import TypedDict, Annotated, Sequence
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from config import settings

# define the state of the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add] # accumales messages over time


# define the nodes
# initialize our tools and model
tool = TavilySearchResults(max_results=2)
model = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.0,
    max_tokens=512,
    groq_api_key=settings.groq_api_key).bind_tools([tool])
logging.BasicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def agent_node(state: AgentState):
    """ 
    the core agent decides wether to use the tool or answer the question"""
    logging.info("...AGENT NODE...")
    response = model.invoke(state['messages'])
    return {"messages": [response]}

def tool_node(state: AgentState):
    """
    Executes the tool call decided by the agent"""
    logging.info("...TOOL NODE...")
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_call = tool_calls[0]
    tool_output = tool.invoke(tool_call["args"])
    return {"messages": [HumanMessage(content=str(tool_output), name="tool")]}

# define the conditional edge
def should_continue(state: AgentState) -> str:
    """
    the router decides wether to continue searching or end """
    logging.info("...ROUTING DECISTION...")
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        logging.info("Decision: END")
        return "end"
    else:
        logging.info("Decision: CONTINUE with tool")
        return "continue"
    
# assemble the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent",should_continue, {"continue": "action",
                                                         "end": END,},)
workflow.add_edge("action", "agent")
#compile the graph in to runnable object
research_app = workflow.compile()