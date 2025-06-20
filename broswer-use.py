import asyncio
import csv
import json
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI

# Define the output format as Pydantic models
class Session(BaseModel):
    Time: str
    SessionType: str
    SessionTitle: str
    PresentationTitle: str
    Authors: str
    Contributors: str
    Location: str

class Sessions(BaseModel):
    sessions: List[Session]

async def main():
    # Create controller with structured output
    controller = Controller(output_model=Sessions)
    
    agent = Agent(
        task="""Extract structured data from "https://s2025.conference-schedule.org/?_gl=1*1xdpc8e*_ga*NTg0NDAzNjI1LjE3NTA0NDAwODI.*_ga_X5ZBLN2D01*czE3NTA0NDAwODEkbzEkZzEkdDE3NTA0NDAzNDAkajU5JGwwJGgw" about all sessions.
                1. Navigate through the calendar page and identify ALL sessions.
                2. For each session, extract the following information:
                   - Time: The time of the session.
                   - SessionType: The type of session such as workshop, tutorial, oral session, poster session, keynote, etc.
                   - SessionTitle: The full title/name of the session.
                   - PresentationTitle: The title of the presentation.
                   - Authors: The list of authors of the session.
                   - Contributors: The list of contributors to the session.
                   - Location: The location of the session.
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
        
        # Save the raw result to a text file
        with open('gtc2025_raw_result.txt', 'w', encoding='utf-8') as f:
            f.write(result)
        
        try:
            # Parse the structured output using Pydantic
            parsed_sessions = Sessions.model_validate_json(result)
            
            # Save to CSV file
            csv_filename = 'gtc2025_sessions_data.csv'
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['date', 'time', 'session_type', 'session_title']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write session data
                for session in parsed_sessions.sessions:
                    writer.writerow({
                        'date': session.date,
                        'time': session.time,
                        'session_type': session.session_type,
                        'session_title': session.session_title
                    })
            
            print(f"Structured data successfully saved to {csv_filename}")
            print(f"Extracted {len(parsed_sessions.sessions)} sessions")
            
            # Print summary statistics
            session_types = {}
            for session in parsed_sessions.sessions:
                session_type = session.session_type
                session_types[session_type] = session_types.get(session_type, 0) + 1
            
            print("\nSession types summary:")
            for session_type, count in session_types.items():
                print(f"  {session_type}: {count} sessions")
            
            # Print first few sessions as examples
            print(f"\nFirst 5 sessions extracted:")
            for i, session in enumerate(parsed_sessions.sessions[:5], 1):
                print(f"\nSession {i}:")
                print(f"  Date: {session.date}")
                print(f"  Time: {session.time}")
                print(f"  Session Type: {session.session_type}")
                print(f"  Session Title: {session.session_title}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON result: {e}")
            print("Attempting fallback parsing...")
            
            # Fallback to manual parsing
            lines = result.strip().split('\n')
            csv_data = []
            
            # Add header
            csv_data.append(['date', 'time', 'session_type', 'session_title'])
            
            for line in lines:
                if line.strip() and (',' in line or '|' in line):
                    if '|' in line:
                        row = [cell.strip() for cell in line.split('|')]
                        row = [cell for cell in row if cell]
                    else:
                        row = [cell.strip() for cell in line.split(',')]
                    
                    if row and len(row) >= 4:
                        csv_data.append(row[:4])  # Only take first 4 columns
            
            if len(csv_data) > 1:  # More than just header
                csv_filename = 'gtc2025_sessions_data_fallback.csv'
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(csv_data)
                
                print(f"Fallback data saved to {csv_filename}")
                print(f"Extracted {len(csv_data)-1} rows of data")
            else:
                print("No structured data found in fallback parsing")
                
        except Exception as e:
            print(f"Error processing structured result: {e}")
    
    else:
        print("No result returned from agent")

asyncio.run(main())