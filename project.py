# ========================= IMPORTING FILES =====================
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict , Annotated,Optional
from pydantic import BaseModel , Field , AnyUrl , EmailStr 
from langchain_core.messages import BaseMessage , SystemMessage , HumanMessage , ToolMessage , AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START , StateGraph , END
import asyncio
from langgraph.prebuilt import tools_condition , ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
import os

# =========================== PYDANTIC MODELS ====================

class company(BaseModel):
    name : str = Field(description = "The name of the company")
    url : str = Field(description="The url of the company")
    role : str = Field(description="The role that company is offering")
    email: str | None = Field(default=None, description="The contact email of the company")
    type : Optional[str] = Field(description="The type of the job whether remote , full time or freelance")

# ### STATE:

class state(TypedDict):
    # Research Agent :
    messages : Annotated[list[BaseMessage] , add_messages]
    Company : company
    route : str
    
# ======================= EXPLICIT PARSER =====================

def explicit_output_parser(data : str , model : type(BaseModel)):
    start = data.find('{')
    end = data.rfind('}') + 1
    if start != -1 and end != 0:
        data = data[start : end]
    data = data.strip()
    if not data.startswith('{'):
        raise ValueError(f"Failed to find a valid JSON object block in model output: {data}")
    return model.model_validate_json(data)

# =============== PROMPT ====================
RESEARCH_PROMPT = SystemMessage(content="""You are a job research assistant.
Use tavily_search tool with ONLY these two parameters:
- query: "AI engineer
- max_results: 1 
Do NOT add any other parameters.

Find a specific individual job posting it contain these items specifically search on Linkedin , Indeed , GlassDoor
From the search results extract:
- company name
- job title/role
- job url (specific job posting not main page)
- job type (remote, full time, freelance)
- email if available (optional)""")

RESEARCH_PARSER_PROMPT = SystemMessage(content="""You are a data extraction assistant.
Extract job information from the search results.
Return ONLY this JSON, no other text:
{
    "name": "<company name>",
    "url": "<specific job posting url>",
    "role": "<job title>",
    "email": null,
    "type": "<remote, full time or freelance>"
}
If email is not found use null.
The url must be a specific job posting, not a listing page.""")

CHAT_BOT_PROMPT = SystemMessage(content="""You are JobBot, a professional AI job search assistant.

Your capabilities:
- Answer questions about careers, resumes, interviews, and job hunting
- Search for real job listings when the user asks
- Give professional advice on job applications

Your behavior:
- Be professional but friendly
- Keep responses concise and helpful
- When user asks to find/search/show jobs, use the search_jobs tool
- For general questions, answer directly without using tools

Rules:
- Never make up job listings
- If you don't know something, say so
- Always encourage the user

You help people find their dream jobs.""")
PLANNER_PROMPT = SystemMessage(content="""You are a query router for a job search assistant.

Your ONLY job is to analyze the user's message and decide where to route it.

If the user wants to:
- Search for jobs
- Find job listings
- Look for vacancies
- Find work opportunities
- Get job recommendations
Reply with exactly one word: SEARCH

If the user wants to:
- Ask a general question
- Get career advice
- Talk about resumes or interviews
- Have a normal conversation
- Anything else
Reply with exactly one word: CHAT

Reply with ONLY "SEARCH" or "CHAT". Nothing else. No explanation.""")


def agent(tools , checkpointer):
    chat_model = ChatGroq(model="llama-3.3-70b-versatile")
    tool_model = chat_model.bind_tools(tools)

    async def planner(state: state):
        result = await chat_model.ainvoke([PLANNER_PROMPT] + state['messages'])
        decision = result.content.strip().upper()
        print(f"Planner decision: {decision}")
        return {'messages': [result], 'route': decision}

    async def researcher(state: state):
        result = await tool_model.ainvoke([RESEARCH_PROMPT] + state['messages'])
        return {'messages': [result]}

    async def researcher_parser(state: state):
        last_message = state['messages'][-1]
        user_content = f"Extract job details from this:\n\n{last_message.content}"
        result = await chat_model.ainvoke([RESEARCH_PARSER_PROMPT, HumanMessage(content=user_content)])
        final_result = explicit_output_parser(result.content, company)
        formatted = f"""
             Company: {final_result.name} \n
             Role: {final_result.role} \n
             Type: {final_result.type} \n
             URL: {final_result.url} \n
             Email: {final_result.email or 'Not provided'}
            """
        return {'Company': final_result, 'messages': [AIMessage(content=formatted)]}

    async def chatbot(state: state):
        result = await chat_model.ainvoke([CHAT_BOT_PROMPT] + state['messages'])
        return {'messages': [result]}

    def route(state: state):
        return state.get('route', 'CHAT')

    tool_node = ToolNode(tools)
    graph = StateGraph(state)

    graph.add_node('planner', planner)
    graph.add_node('researcher', researcher)
    graph.add_node('researcher_parser', researcher_parser)
    graph.add_node('tool_node', tool_node)
    graph.add_node('chatbot', chatbot)
    graph.add_edge(START, 'planner')
    graph.add_conditional_edges('planner', route, {
        'SEARCH': 'researcher',
        'CHAT': 'chatbot'
    })
    graph.add_conditional_edges('researcher', tools_condition, {
        'tools': 'tool_node',
        '__end__': 'researcher_parser'
    })
    graph.add_edge('tool_node', 'researcher')

    # end nodes
    graph.add_edge('researcher_parser', END)
    graph.add_edge('chatbot', END)

    return graph.compile(checkpointer = checkpointer)
        


async def main():
    client = MultiServerMCPClient({
        "tavily": {
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}",
            "transport": "streamable_http"
        }
    })
    
    tavily_tools = await client.get_tools()
    async with AsyncSqliteSaver.from_conn_string("jobbot.db") as checkpointer:
        app = agent(tavily_tools, checkpointer)
        print("JobBot ready! Type 'quit' to exit.\n")
        thread_id = input("Enter your name or ID to load your chat: ")
        config = {"configurable": {"thread_id": thread_id}}
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            print("JobBot: ", end="", flush=True)
            async for chunk in app.astream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates"
            ):
                for node_name, values in chunk.items():
                    if 'messages' in values:
                        last = values['messages'][-1]
                        if hasattr(last, 'content') and last.content:
                            if node_name in ['chatbot', 'researcher_parser']:
                                print(last.content, end="", flush=True)
            print("\n")

if __name__ == "__main__":
    asyncio.run(main())
