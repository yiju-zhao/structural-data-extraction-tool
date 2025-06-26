import asyncio
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, field_validator

from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI


class Fields(BaseModel):
	Title: str
	Doi: str
	Author: str
	Affiliation: str
	
	@field_validator('Author', 'Affiliation', mode='before')
	@classmethod
	def convert_list_to_string(cls, v):
		"""Convert list to comma-separated string if input is a list."""
		if isinstance(v, list):
			return ', '.join(str(item) for item in v)
		return str(v) if v is not None else "N/A"


class Papers(BaseModel):
	papers: list[Fields]
	
controller = Controller(output_model=Papers)

async def main():
	# Updated task to instruct the agent NOT to create files, only return JSON
	task = (
	    'Navigate to https://kdd2025.kdd.org/research-track-papers-2/#august_cycle and click "August Cycle" to extract ALL "August Cycle" papers. '
	    'For each paper, capture these fields exactly: Title, Doi, Author, Affiliation. '
	    'Scroll until the bottom of the page and return the complete dataset from the beginning to the end of the page in JSON format matching this schema: '
	    '{"papers": [{"Title": "", "Doi": "", "Author": "", "Affiliation": ""}, ...]}. '
	    'Do NOT create or write any files—simply return the JSON via the done action. '
	    'If a data point is missing, use "N/A".'
	)
	model = ChatOpenAI(model='gpt-4.1')
	agent = Agent(task=task, llm=model, controller=controller)

	history = await agent.run()

	result = history.final_result()
	if result:
		parsed: Papers = Papers.model_validate_json(result)

		# Save to JSON file
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = f"kdd2025_august_papers_{timestamp}.json"
		
		with open(filename, 'w', encoding='utf-8') as f:
			json.dump(parsed.model_dump(), f, indent=2, ensure_ascii=False)
		
		print(f"Data saved to {filename}")
		print(f"Total papers extracted: {len(parsed.papers)}")

		for post in parsed.papers:
			print('\n--------------------------------')
			print(f'Title:            {post.Title}')
			print(f'Doi:              {post.Doi}')
			print(f'Author:           {post.Author}')
			print(f'Affiliation:      {post.Affiliation}')
	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())