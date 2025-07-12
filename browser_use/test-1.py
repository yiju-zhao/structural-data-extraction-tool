from browser_use import Agent  
from browser_use.browser import BrowserSession  
from browser_use.llm import ChatOpenAI  

import asyncio
import pathlib
import os
import shutil

SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
agent_dir = SCRIPT_DIR / 'test_no_thinking'
agent_dir.mkdir(exist_ok=True)
conversation_dir = agent_dir / 'conversations' / 'conversation'
print(f'Agent logs directory: {agent_dir}')

# Initialize your components  
llm = ChatOpenAI(model="gpt-4.1")  # or your preferred LLM 
url = 'https://s2025.conference-schedule.org'  # URL to scrape

agent = Agent(  
    task=f"""  
    1. Navigate to the {url}, expand all expandable sections.
    2. Save session information data to results.md on the pageview.
    3. Scroll down to next page. (Make sure to not scroll too much, just enough to load the next page)
    4. Use append_file to add findings on the new pageview to results.md. 
    5. Repeat steps 3-4 until no more pages are available.

    NOTE: DO NOT USE extract_structured_data action - everything is visible in browser state.
    """,  
    llm=llm,
    save_conversation_path=str(conversation_dir),
	file_system_path=str(agent_dir / 'fs'),
)


async def main():
# Create agent with a task that includes scrolling and data saving  
    # Run the agent  
    history = await agent.run()
    print(f"Final result: {history.final_result()}", flush=True)

if __name__ == '__main__':
	asyncio.run(main())