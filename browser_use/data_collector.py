"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys
import pathlib
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

from browser_use import Agent, Controller
from browser_use.llm import ChatOpenAI
from utility.json_to_csv_converter import convert_browser_use_output_to_csv


class Session(BaseModel):
	date: str
	time: str
	type: str
	session_presentation_title: str
	contributors: list[str]
	location: str

class Sessions(BaseModel):
	sessions: list[Session]


SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
agent_dir = SCRIPT_DIR / 'output'
agent_dir.mkdir(exist_ok=True)

controller = Controller(output_model=Sessions)


async def main():
	url = 'https://iccv.thecvf.com/virtual/2025/events/tutorial'
	# You can also use a local file, e.g. 'file:///path/to/file
	task = f'Go to {url} and extract all tutorial metadata using extract_structured_data.'
	model = ChatOpenAI(model='gpt-5')
	planner_model = ChatOpenAI(model='o4-mini', temperature=1)
	agent = Agent(task=task, llm=model, controller=controller, file_system_path=str(agent_dir))

	history = await agent.run()

	result = history.final_result()
	if result:

		# Save to CSV according to the Pydantic model
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		csv_path = agent_dir / f"conference_sessions_{timestamp}.csv"
		try:
			convert_browser_use_output_to_csv(
				raw_result=str(result),
				pydantic_model=Sessions,
				csv_filename=str(csv_path),
				preserve_all_data=True,
			)
			print(f"CSV saved: {csv_path}")
		except Exception as e:
			print(f"Failed to save CSV: {e}")
	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())
