from textwrap import dedent
from typing import Iterator
from fastapi import FastAPI
from agno.agent import Agent, RunOutputEvent
from agno.models.openai import OpenAIResponses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://wp-test.local", "https://shop.greenshiftwp.com", "https://greenshiftwp.com"],  # Allows specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5-mini"),
    tools=[{"type": "web_search_preview"}],
    description=dedent("""
        You are an expert search agent that queries the site greenshiftwp.com.
        Based on the user's request, you attempt to find relevant links on the site.
        If no results are found, you respond with 'I was unable to find any results on Greenshift Documentation. Please write to support.'.
        If links are found, ensure they are returned at the end of your response.
        Additionally, extract content from the top two relevant links to provide deeper insights.
    """),
    instructions=dedent("""
        Follow these steps for each search request:

        1. Search Phase 🔍
           - Use web search preview to search the site greenshiftwp.com based on the user's query.
           - Check if any links are available after the search.
           - Filter to the top two most relevant links.

        2. Content Extraction 📰
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
    add_history_to_context=True,
    add_datetime_to_context=True,
    num_history_sessions=3
)

# Helper function to make objects JSON serializable
def make_serializable(obj):
    """Convert non-serializable objects to serializable form."""
    if hasattr(obj, '__dict__'):
        return {k: make_serializable(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        try:
            # Try standard JSON serialization
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # If not serializable, convert to string representation
            return str(obj)

@app.get("/ask")
async def ask(query: str):
    response = agent.run(query)
    return {"response": response.content}

@app.get("/ask-with-stream")
async def ask_with_stream(query: str):
    # Run agent and return the response as a stream
    response_stream: Iterator[RunOutputEvent] = agent.run(query, stream=True)
    
    def generate():
        # Add a flag to track if we've already sent the tools data
        tools_sent = False
        
        for chunk in response_stream:
            # If there's tool call information and we haven't sent it yet
            if hasattr(chunk, "tools") and chunk.tools and not tools_sent:
                # Convert tools to JSON serializable format
                serializable_tools = make_serializable(chunk.tools)
                data = {
                    "tools": serializable_tools,
                    "type": "tools"
                }
                yield f"data: {json.dumps(data)}\n\n"
                # Mark that we've sent the tools data
                tools_sent = True
            
            # Each chunk is a RunResponse object
            # Format as Server-Sent Event
            if chunk.content is not None:
                data = {
                    "content": chunk.content,
                    "type": "content"
                }
                yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/healthz/")
def health_check_endpoint():
    return {"status": "ok"}