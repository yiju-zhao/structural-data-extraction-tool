"""
Browser Use Agent for Web Analysis and Python Script Generation

This module uses browser_use agent to analyze web pages and generate Python scripts 
for data extraction based on the structure defined in session.json.
"""

import asyncio
import argparse
import json
import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from browser_use import Agent
from browser_use.llm import ChatOpenAI


class ScraperWriter:
    def __init__(self, session_config_path: str = None):
        """Initialize the scraper writer with session configuration."""
        self.session_config_path = session_config_path or "/Users/eason/Documents/HW Project/Agent/Tools/structural-data-extraction-tool/browser_use/session.json"
        self.session_config = self._load_session_config()
        self.model = ChatOpenAI(model='gpt-4.1', temperature=0)
        self.planner_model = ChatOpenAI(model='o4-mini', temperature=1)
    
    def _load_session_config(self) -> Dict[str, Any]:
        """Load the session configuration from JSON file."""
        try:
            with open(self.session_config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Session config file not found: {self.session_config_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Invalid JSON in session config file: {self.session_config_path}")
            return {}
    
    def _generate_script_prompt(self, url: str) -> str:
        """Generate a detailed prompt for the agent to analyze the web page and create a Python script."""
        
        # Extract field information from session config
        target_columns = []
        if self.session_config:
            for item in self.session_config:
                if 'attributes' in item:
                    for attr in item['attributes']:
                        target_columns.append(attr['name'])
        
        columns_text = ", ".join(target_columns) if target_columns else "No specific columns defined"
        
        prompt = f"""
You are a web-scraping agent with full browsing capability. Your goal is to generate a Python script that will automatically extract table data from a webpage.

TARGET URL: {url}
TARGET COLUMNS: {columns_text}

Your task is to:
1. Navigate to the provided URL: {url}
2. Locate any HTML table(s) on the page
3. Identify which table contains all of the requested columns: {columns_text}
4. Extract only those columns into a pandas DataFrame
5. Write the DataFrame to a CSV file named 'output.csv'

Generate a standalone Python script that:
- Uses appropriate libraries (requests, BeautifulSoup, pandas, or pandas.read_html)
- Includes comprehensive error handling for cases where:
  * The URL is unreachable
  * No tables are found on the page
  * None of the tables contain the required columns
  * Network or parsing errors occur
- Implements robust table detection and column matching logic
- Saves the extracted data as 'output.csv'
- Includes clear logging/print statements to show progress
- Has proper imports and dependencies listed at the top

The script should be production-ready with:
- Clear comments explaining each step
- Data validation and cleaning
- Graceful error handling with informative messages
- Main execution block

Please provide ONLY the complete Python script code, no additional explanation or markdown formatting.
"""
        return prompt
    
    async def analyze_and_generate_script(self, url: str, output_file: str = None) -> str:
        """
        Analyze the web page and generate a Python script for data extraction.
        
        Args:
            url: The URL to analyze
            output_file: Optional path to save the generated script
            
        Returns:
            The generated Python script as a string
        """
        
        task = self._generate_script_prompt(url)
        
        # Create agent with the task
        agent = Agent(task=task, llm=self.model, planner_llm=self.planner_model, planner_interval=2)
        
        print(f"Starting analysis of {url}...")
        print("This may take a few minutes as the agent analyzes the page structure...")
        
        # Run the agent
        history = await agent.run()
        
        # Get the final result
        result = history.final_result()
        
        if result:
            # Save the generated script if output file is specified
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(result)
                print(f"Generated script saved to: {output_file}")
            
            return result
        else:
            print("No script was generated")
            return ""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Python scraper scripts using browser_use agent")
    
    parser.add_argument("--config", 
                       default="/Users/eason/Documents/HW Project/Agent/Tools/structural-data-extraction-tool/browser_use/session.json",
                       help="Path to session config JSON file")
    
    parser.add_argument("--url", 
                       required=True,
                       help="URL to analyze and scrape")
    
    parser.add_argument("--output", 
                       help="Output file path for the generated script")
    
    return parser.parse_args()


async def main():
    """Main function to run the scraper writer."""
    
    args = parse_args()
    
    # Generate default output filename if not provided
    if not args.output:
        domain = args.url.split("//")[-1].split("/")[0]
        # Sanitize domain name to be alphanumeric only
        sanitized_domain = ''.join(c for c in domain if c.isalnum() or c == '_')
        if not sanitized_domain:
            sanitized_domain = "website"
        args.output = f"scraper_{sanitized_domain}.py"
    
    print(f"Using config: {args.config}")
    print(f"Analyzing URL: {args.url}")
    print(f"Output script: {args.output}")
    
    # Create scraper writer instance with session config path
    scraper_writer = ScraperWriter(session_config_path=args.config)
    
    # Generate the script
    script = await scraper_writer.analyze_and_generate_script(args.url, args.output)
    
    if script:
        print("\n" + "="*50)
        print("GENERATED SCRIPT:")
        print("="*50)
        print(script)
        print("="*50)
    else:
        print("Failed to generate script")


if __name__ == '__main__':
    asyncio.run(main())