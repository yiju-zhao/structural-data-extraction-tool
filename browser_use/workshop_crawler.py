import asyncio
import sys
import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
from browser_use import Agent, Controller
from browser_use.llm import ChatOpenAI
#from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.json_to_csv_converter import convert_browser_use_output_to_csv


# Define the output format as Pydantic models
class Session(BaseModel):
    Title: str
    Abstract: str
    Speakers: str


class Sessions(BaseModel):
    sessions: List[Session]


async def main():
    # Create controller with structured output
    controller = Controller(output_model=Sessions)

    agent = Agent(
        task="""Extract structured information about each workshop listed on https://colmweb.org/workshops.html. 

                Navigate through the workshop page and identify ALL workshops.
                For each workshop, visit the workshop website and extract the following information:

                1. Title: Title of the workshop
                2. Abstract: The abstract of the workshop introducing the workshop
                3. Speakers: All the invited speakers' names

                Only include invited speakers—do not list program chairs, organizing committee, etc., unless they are clearly workshop invited speakers.
                If any data points are missing, mark them as "N/A" rather than leaving them blank.
                Be concise and extract only the necessary structured information. Skip workshops if their external website is broken or lacks desired information.""",
        llm=ChatOpenAI(model="gpt-4.1"),
        #llm=ChatOpenAI(model="gpt-4.1-mini", temperature=1.0),
        controller=controller,
    )

    # Run the agent and capture the history
    history = await agent.run()

    # Get the final result from the agent
    result = history.final_result()

    if result:
        print("Agent completed successfully!")
        print("Raw result:")
        print(result)

        # Save the complete raw result to a text file - ensure we capture everything
        raw_result_str = str(result)  # Ensure it's a string
        print(f"\n📝 Saving raw result ({len(raw_result_str)} characters) to file...")

        with open('colm_workshop_raw_result.txt', 'w', encoding='utf-8') as f:
            f.write(raw_result_str)
            f.flush()  # Ensure data is written to disk

        print(f"✅ Raw result saved to colm_workshop_raw_result.txt")

        # Convert JSON result to CSV using the utility function with NO filtering
        csv_filename = 'colm_workshop_results.csv'
        print(f"\n🔄 Converting to CSV with complete data preservation...")

        # Use the converter but disable aggressive filtering to preserve all data
        stats = convert_browser_use_output_to_csv(
            raw_result=raw_result_str,
            pydantic_model=Sessions,
            csv_filename=csv_filename,
            preserve_all_data=True  # New parameter to disable filtering
        )

        print(f"\n✅ CSV file saved successfully: {csv_filename}")

    else:
        print("No result returned from agent")


asyncio.run(main())
