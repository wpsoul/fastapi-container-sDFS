from textwrap import dedent
from fastapi import FastAPI
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from fastapi.middleware.cors import CORSMiddleware

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
        If no results are found, you respond with 'sorry, I have no results'.
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

@app.get("/healthz/")
def health_check_endpoint():
    return {"status": "ok"}