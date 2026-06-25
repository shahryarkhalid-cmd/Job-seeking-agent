from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage
import asyncio
import os
import json
from project import agent
from langchain_mcp_adapters.client import MultiServerMCPClient
app = FastAPI()

class ChatRequest(BaseModel):
    thread_id: str
    message: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()

@app.post("/chat")
async def chat(request: ChatRequest):
    async def stream():
        # ✅ no async with
        client = MultiServerMCPClient({
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()

        async with AsyncSqliteSaver.from_conn_string("jobbot.db") as checkpointer:
            app_agent = agent(tools, checkpointer)
            config = {"configurable": {"thread_id": request.thread_id}}

            async for chunk in app_agent.astream(
                {'messages': [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="updates"
            ):
                for node_name, values in chunk.items():
                    if 'messages' in values:
                        last = values['messages'][-1]
                        if hasattr(last, 'content') and last.content:
                            if node_name in ['chatbot', 'researcher_parser']:
                                yield f"data: {json.dumps({'text': last.content})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)