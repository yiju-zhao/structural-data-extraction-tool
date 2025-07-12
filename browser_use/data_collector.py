"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

from browser_use import Agent, Controller
from browser_use.llm import ChatOpenAI


class Session(BaseModel):
	date: str
	time: str
	type: str
	session_presentation_title: str
	contributors: list[str]
	location: str

class Sessions(BaseModel):
	sessions: list[Session]


controller = Controller(output_model=Sessions)


async def main():
	url = 'https://s2025.conference-schedule.org'
	# You can also use a local file, e.g. 'file:///path/to/file
	task = f'Go to {url} and extract top 70 session using extract_structured_data.'
	model = ChatOpenAI(model='gpt-4.1')
	planner_model = ChatOpenAI(model='o4-mini', temperature=1)
	agent = Agent(task=task, llm=model, controller=controller)

	history = await agent.run()

	result = history.final_result()
	if result:
		parsed: Sessions = Sessions.model_validate_json(result)

		for session in parsed.sessions:
			print('\n--------------------------------')
			print(f'Title:            {session.session_presentation_title}')
			print(f'date:             {session.date}')
			print(f'time:             {session.time}')
			print(f'type:             {session.type}')
			print(f'contributors:     {", ".join(session.contributors)}')
			print(f'location:         {session.location}')
			print('--------------------------------\n')
	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())