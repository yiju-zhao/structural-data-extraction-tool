from browser_use import Agent, Controller
from browser_use.browser import BrowserSession  
from browser_use.llm import ChatOpenAI  
from pydantic import BaseModel
import asyncio
import pathlib
import os
import shutil

SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
agent_dir = SCRIPT_DIR / 'output'
agent_dir.mkdir(exist_ok=True)
conversation_dir = agent_dir / 'conversations'
print(f'Agent logs directory: {agent_dir}')


url = 'https://s2025.conference-schedule.org'  # URL to scrape
tab = 'Monday'  # Tab to click
cleaned_url = url.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '')  # Cleaned URL for file naming

class Session(BaseModel):
	time: str
	type: str
	contributors: list[str]

class Sessions(BaseModel):
	sessions: list[Session]

controller = Controller(output_model=Sessions)

prompt = f"""

        ## Objective
        Extract all conference session details from {url}/{tab} and save them to {cleaned_url}_{tab}.md in a structured markdown format.

        ## Workflow

        ### 1. Initial Setup
        - CREATE {cleaned_url}_{tab}.md
        - OPEN [{url}]
        - CLICK [{tab}]
        - EXPAND ALL collapsible sections

        ### 2. Extraction Process (Repeat Until Completion)
        For each session:
        1. EXTRACT:
        - Session/presentation title
        - Time information
        - Session type
        - Contributors list

        2. WRITE TO FILE (append_file action):
        ## session_presentation_title: <title>
        - time: <time>
        - type: <type>
        - contributors: <contributor1>, <contributor2>, ...

        3. PAGE NAVIGATION:
        - SCROLL DOWN to load more sessions
        - CONTINUE until "Chapters Party" session or page bottom

        ### 3. Completion Criteria
        - STOP when:
        a) "Chapters Party" session is found, OR
        b) Page bottom is reached

        ## Restrictions
        - ❌ NEVER use extract_structured_data()
        - ❌ NEVER terminate early
        - ❌ NEVER skip sessions
        """
# Initialize your components  
llm = ChatOpenAI(model="gpt-4.1")  # or your preferred LLM 

agent = Agent(  
    task=prompt,  
    llm=llm,
    # controller=controller,
    # save_conversation_path=str(conversation_dir),
	file_system_path=str(agent_dir),
)


async def main():
# Create agent with a task that includes scrolling and data saving  
    # cleanup conversations directory if it exists
    if conversation_dir.exists():
        shutil.rmtree(conversation_dir)
    # Run the agent  
    await agent.run()

if __name__ == '__main__':
	asyncio.run(main())