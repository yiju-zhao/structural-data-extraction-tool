from browser_use import Agent, Controller
from browser_use.browser import BrowserSession, BrowserProfile
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

google_sheet_url = 'https://docs.google.com/spreadsheets/d/1jO54spKGDXePbv-eBRcORXdefL3y8-tIZEjNU2WUzB0/edit?usp=sharing'  

batch_size = 3  # Number of sessions to process in one go

class Session(BaseModel):
	time: str
	type: str
	session_presentation_title: str
	contributors_authors: list[str]

class Sessions(BaseModel):
	sessions: list[Session]

controller = Controller(output_model=Sessions)

# Initialize your components  
llm = ChatOpenAI(model="gpt-4.1")  # or your preferred LLM 
planner_model = ChatOpenAI(model='o4-mini', temperature=1)

prompt = f"""  
            # Goal: Extract all conference sessions from {url}/{tab} → Append to Google Sheet {google_sheet_url}

            ## INITIAL SETUP
            1. OPEN [Google Sheet: {google_sheet_url}] in new tab
            - WRITE ROW 1: ["session_presentation_title", "time", "type", "contributors"]

			2. OPEN [{url}] in new tab 
			    - CLICK [{tab}]
				- EXPAND ALL COLLAPSIBLE SECTIONS

            ## EXTRACTION LOOP: REPEAT UNTIL END OF PAGE
            BEGIN LOOP:
            
            1. EXTRACTION:
                - SWITCH to {url} tab
				- INSPECT webpage to find visible sessions
                - WRITE the sessions info data to {cleaned_url}_temp.md according to the output_model
            
            2. APPEND TO SHEET:
                - SWITCH to Google Sheet tab
                - FIND next empty row in Sheet
				- READ {cleaned_url}_temp.md content
                - WRITE values to respective columns
				- MATCH columns:
                    - A: session_presentation_title
                    - B: time
                    - C: type
                    - D: contributors_authors
            
            3. CLEANUP:
                - CLENN {cleaned_url}_temp.md content
            
			4. RETURN to {url} tab
                - SCROLL DOWN to load more sessions
                - REPEAT LOOP until no more pixels can be scrolled down
			
			END LOOP

            ## PROHIBITED ACTIONS
            ❌ NEVER use extract_structured_data()
            ❌ NEVER overwrite existing sheet data
            ❌ NEVER terminate before reaching webpage bottom
            """


async def main():
# Create agent with a task that includes scrolling and data saving  
    # cleanup conversations directory if it exists
    if conversation_dir.exists():
        shutil.rmtree(conversation_dir)
		
    agent = Agent(  
        task=prompt,  
        llm=llm,
		# planner_llm=planner_model,
		# planner_interval= 5,
        controller=controller,
        # save_conversation_path=str(conversation_dir),
        file_system_path=str(agent_dir),
    )
    # Run the agent  
    await agent.run()

if __name__ == '__main__':
	asyncio.run(main())