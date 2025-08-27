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
tab = 'THURSDAY'  # Tab to click
cleaned_url = url.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '')  # Cleaned URL for file naming

class Session(BaseModel):
	time: str
	type: str
	contributors: list[str]

class Sessions(BaseModel):
	sessions: list[Session]

controller = Controller(output_model=Sessions)

prompt = f"""
        Objective

        Extract all conference session details from {url}/{tab} and save them to {cleaned_url}_{tab}.md in structured markdown
        format.

        Pre-execution Setup

        1. File Creation: CREATE {cleaned_url}_{tab}.md with header:
        # {tab} Sessions - {cleaned_url}

        2. Browser Navigation:
            - NAVIGATE to [{url}]
            - WAIT for page load (3-5 seconds)
            - LOCATE and CLICK [{tab}] tab/button
            - WAIT for content load
            - EXPAND ALL collapsible/accordion sections

        Session Extraction Protocol

        Detection Strategy

        - SCAN page for session containers (divs, cards, list items)
        - IDENTIFY unique session markers (titles, time stamps, speaker names)
        - SET scroll position tracker to avoid re-processing

        Extraction Loop

        For EACH detected session:

        1. Data Capture:
        title: [session/presentation title - exact text]
        time: [time/date info - format as found]
        type: [session type/category if available]
        contributors: [speaker1, speaker2, ... - comma separated]
        2. File Write (append_file):
        ## session.title
        - **Time**: session.time
        - **Type**: session.type
        - **Contributors**: session.contributors

        3. Progress Tracking:
            - MARK session as processed
            - INCREMENT session counter
            - LOG current position

        Navigation Control

        - SCROLL DOWN incrementally (viewport height)
        - WAIT 2-3 seconds for dynamic loading
        - CHECK for new sessions after each scroll
        - STOP when "Chapters Party" session found AND saved

        Validation & Completion

        - Session Count: Log total sessions extracted
        - End Marker: Confirm "Chapters Party" session captured
        - File Check: Verify markdown file exists and contains data
        - No Duplicates: Ensure each session appears only once

        Error Recovery

        - If page load fails: RETRY once, then REPORT error
        - If tab not found: SEARCH for similar text, then REPORT
        - If no sessions detected: SCROLL and wait, retry detection
        - If stuck in loop: CHECK scroll position change, break if no progress

        Success Criteria

        ✅ All sessions from page start to "Chapters Party" extracted
        ✅ Structured markdown file created
        ✅ No duplicate entries
        ✅ No infinite loops or early termination
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