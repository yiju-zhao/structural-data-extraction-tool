import asyncio
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI
from json_to_csv_converter import convert_browser_use_output_to_csv

# Define the output format as Pydantic models
class Session(BaseModel):
    Title: str
    Author: str
    Affiliation: str
    Doi: str

class Sessions(BaseModel):
    sessions: List[Session]

async def main():
    # Create controller with structured output
    controller = Controller(output_model=Sessions)
    
    agent = Agent(
        task="""Extract structured data from "https://kdd2025.kdd.org/research-track-papers-2/" about all sessions.
                1. Navigate through the calendar page and identify ALL sessions.
                2. For each session, extract the following information:
                   - Title: The title of the paper.
                   - Author: The author of the paper.
                   - Affiliation: The affiliation of the author.
                   - Doi: The DOI of the paper.
                3. Make sure to capture ALL sessions across all days of the conference.
                4. If any data points are missing, mark them as "N/A" rather than leaving them blank.
                5. Preserve the chronological order of sessions.""",
        llm=ChatOpenAI(model="gpt-4.1-mini"),
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

        with open('kdd_paper_raw_result.txt', 'w', encoding='utf-8') as f:
            f.write(raw_result_str)
            f.flush()  # Ensure data is written to disk

        print(f"✅ Raw result saved to kdd_paper_raw_result.txt")
        
        # Convert JSON result to CSV using the utility function with NO filtering
        csv_filename = 'kdd_paper_results.csv'
        print(f"\n🔄 Converting to CSV with complete data preservation...")

        # Use the converter but disable aggressive filtering to preserve all data
        stats = convert_browser_use_output_to_csv(
            raw_result=raw_result_str,
            pydantic_model=Sessions,
            csv_filename=csv_filename,
            preserve_all_data=True  # New parameter to disable filtering
        )

        # Additional analysis if conversion was successful
        if stats and stats.get('valid_items_written', 0) > 0:
            try:
                # Parse for additional statistics
                parsed_sessions = Sessions.model_validate_json(raw_result_str)

                print(f"\n📊 Data Summary:")
                print(f"   Total papers extracted: {len(parsed_sessions.sessions)}")

                # Count papers by affiliation if available
                affiliations = {}
                for session in parsed_sessions.sessions:
                    if hasattr(session, 'Affiliation') and session.Affiliation and session.Affiliation != "N/A":
                        affiliation = session.Affiliation
                        affiliations[affiliation] = affiliations.get(affiliation, 0) + 1

                if affiliations:
                    print(f"\n📈 Top 10 Affiliations:")
                    sorted_affiliations = sorted(affiliations.items(), key=lambda x: x[1], reverse=True)[:10]
                    for affiliation, count in sorted_affiliations:
                        print(f"   {affiliation}: {count} papers")

            except Exception as analysis_error:
                print(f"⚠️  Additional analysis failed: {analysis_error}")

        print(f"\n✅ CSV file saved successfully: {csv_filename}")
    
    else:
        print("No result returned from agent")

asyncio.run(main())