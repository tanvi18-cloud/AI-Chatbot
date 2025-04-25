from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from instructions import information
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse

# ✅ Directly adding API key instead of using .env
OPENAI_API_KEY = "your_openai_api_key_here"

# Define a custom fallback tool for unknown queries
@tool(description="Called when query falls outside your knowledge base")
def web_search(query: str) -> str:
    try:
        # CHARUSAT's official website URL
        base_url = "https://charusat.ac.in"
        
        # Encode the query for URL
        encoded_query = urllib.parse.quote(query)
        
        # Try searching the website's search page
        search_url = f"{base_url}/?s={encoded_query}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for search results
        results = soup.find_all(['article', 'div', 'section'], class_=['post', 'entry', 'content'])
        
        if results:
            # Get the first relevant result
            for result in results:
                text = result.get_text().strip()
                if text and len(text) > 50:  # Ensure we have meaningful content
                    return f"I found this information on CHARUSAT's website: {text[:500]}... [Read more at charusat.ac.in]"
        
        # If no results found, try specific pages
        specific_pages = {
            "startup": "/innovation-incubation-center/",
            "research": "/research/",
            "student": "/student-life/",
            "faculty": "/faculty/",
            "admission": "/admission/",
            "placement": "/placement/",
            "international": "/international/",
            "library": "/library/",
            "sports": "/sports/",
            "campus": "/campus-life/"
        }
        
        # Check if query matches any specific page
        for keyword, page in specific_pages.items():
            if keyword in query.lower():
                page_url = f"{base_url}{page}"
                page_response = requests.get(page_url)
                if page_response.status_code == 200:
                    page_soup = BeautifulSoup(page_response.text, 'html.parser')
                    content = page_soup.find('main') or page_soup.find('article') or page_soup.find('div', class_='content')
                    if content:
                        text = content.get_text().strip()
                        if text and len(text) > 50:
                            return f"I found this information on CHARUSAT's website: {text[:500]}... [Read more at {page_url}]"
        
        return "I couldn't find specific information about this on CHARUSAT's website. Please visit charusat.ac.in or contact the university directly for more information."
    except Exception as e:
        return f"I tried to search CHARUSAT's website but encountered an error: {str(e)}. Please try again later or visit charusat.ac.in directly."

tools = [web_search]

# ✅ Initialize LangChain AI Client
client = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.4, 
    api_key="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" # Using the direct API key
)

# ✅ Define chatbot prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", information),
    ("placeholder", "{messages}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ✅ Create chatbot agent
agent = create_tool_calling_agent(client, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ✅ Temporary chat history storage
demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="messages",
    output_messages_key="output",
)

# ✅ Function to process user messages
def run_agent(messages, session_id):
    response = conversational_agent_executor.invoke(
        {"messages": [messages]},
        {"configurable": {"session_id": session_id}},
    )   
    return response
