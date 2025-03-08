from textwrap import dedent
from typing import Iterator
from fastapi import FastAPI
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://wp-test.local"],  # Allows specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

agent = Agent(
    name="GreenShiftSearchAgent",
    tools=[
        DuckDuckGoTools(modifier="site:greenshiftwp.com"),  # Search only on greenshiftwp.com
        Newspaper4kTools(),  # For extracting content from the links
    ],
    model=OpenAIChat(id="gpt-4o-mini"),  # Using gpt-4o-mini for better performance
    description=dedent("""
        You are an expert search agent that queries the site greenshiftwp.com using DuckDuckGo.
        Based on the user's request, you attempt to find relevant links on the site.
        If no results are found, you respond with 'I was unable to find any results on Greenshift Documentation. Please write to support.'.
        If links are found, ensure they are returned at the end of your response.
        Additionally, extract content from the top two relevant links to provide deeper insights.
    """),
    instructions=dedent("""
        Follow these steps for each search request:

        1. Search Phase 🔍
           - Use DuckDuckGo to search the site greenshiftwp.com based on the user's query.
           - Check if any links are available after the search.
           - Filter to the top two most relevant links.

        2. Content Extraction 📰
           - Use Newspaper4kTools to extract content from the top two links.
           - Provide deeper insights based on the extracted content.

        3. Response Process 📄
           - If no results are found, respond with 'sorry, I have no results'.
           - If links are found, ensure they are included at the end of your response.

        Remember:
        - Only search the site greenshiftwp.com
        - Return links if available, otherwise provide a no results message.
        - Extract and utilize content from the top two links for deeper answers.
    """),
    markdown=True,
    add_datetime_to_instructions=True,
    show_tool_calls=False,
)


@app.get("/ask")
async def ask(query: str):
    response = agent.run(query)
    return {"response": response.content}

@app.get("/ask-with-stream")
async def ask_with_stream(query: str):
    # Run agent and return the response as a stream
    response_stream: Iterator[RunResponse] = agent.run(query, stream=True, show_tool_calls=True)
    
    def generate():
        for chunk in response_stream:
            # Each chunk is a RunResponse object
            # Format as Server-Sent Event
            if chunk.content is not None:
                data = {
                    "content": chunk.content,
                    "type": "content"
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            # If there's tool call information and show_tool_calls is enabled
            if hasattr(chunk, "tools") and chunk.tools:
                data = {
                    "tools": chunk.tools,
                    "type": "tools"
                }
                yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/healthz/")
def health_check_endpoint():
    return {"status": "ok"}