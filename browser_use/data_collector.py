import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from browser_use import Agent, Controller
from browser_use.llm import ChatOpenAI

load_dotenv()

controller = Controller()

class Config(BaseModel):
    """
    Application configuration loaded from environment or arguments.
    """
    url: HttpUrl
    entity_type: str
    output_dir: Path = Path("output")

    @classmethod
    def from_args_and_env(cls, entity_type: str, url: str = None) -> "Config":
        """Load configuration from arguments and environment variables."""
        if not url:
            raise ValueError("URL must be provided via argument or TARGET_URL environment variable")
        
        output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        
        return cls(url=url, entity_type=entity_type, output_dir=output_dir)

# Global configuration (initialized in main)
config = None

class DataEntity(BaseModel):
    """Generic model for extracted data entities."""
    title: str
    date: str
    time: str
    type: str
    contributors: List[str]
    location: str

class DataEntities(BaseModel):
    """Collection of data entities."""
    entities: List[DataEntity]
    entity_type: str 


@controller.action(description="Save Extracted Data", param_model=DataEntities)
def save_data(data_entities: DataEntities) -> None:
    """Save extracted data to JSON file."""
    try:
        config.output_dir.mkdir(exist_ok=True)
        entity_type_clean = config.entity_type.replace("/", "_").replace(" ", "_")
        file_path = config.output_dir / f"{entity_type_clean}.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            for entity in data_entities.entities:
                f.write(f"Title: {entity.title}")
                f.write(f"Date: {entity.date}")
                f.write(f"Time: {entity.time}")
                f.write(f"Type: {entity.type}")
                f.write(f"Contributors: {', '.join(entity.contributors)}")
                f.write(f"Location: {entity.location}")
                f.write("\n---\n\n")
        
        print(f"{config.entity_type.title()} data saved to {file_path}")
    except Exception as e:
        print(f"Error saving {config.entity_type}: {e}")
        raise

async def main(entity_type: str, url: str = None) -> None:
    """Main function to run the data extraction."""
    global config
    config = Config.from_args_and_env(entity_type, url)
    
    task = f"""
        Extract ALL {config.entity_type} information from: {config.url}

        **Step-by-Step Instructions:**

        1. **Load and Explore the Page:**
        - Navigate to the URL and let it fully load
        - Examine the page layout to understand how content is organized

        2. **Scroll Through the ENTIRE Page:**
        - Fastly scroll down from top to bottom
        - Look for pagination buttons, "Load More" buttons, or infinite scroll
        - Click any "Load More" or pagination buttons to reveal additional content
        - Continue scrolling until you reach the absolute bottom with no new content

        Go back to the top of the page after scrolling to ensure you have loaded all content.
        3. **Extract ALL Data:**
        - After scrolling through the entire page and loading all content, collect every single {config.entity_type} item you can find
        - For each item, extract all available information including:
            • Title/Name
            • Date and Time (if available)
            • Type/Category
            • Contributors/Authors/Speakers
            • Location/Venue
            • Any other relevant details you can see

        4. **Save All Data at Once:**
        - After collecting ALL items from the complete page, save everything using the save_data() function
        - Do NOT save in batches - save all items together in one operation

        **Important:** You must scroll through and load the ENTIRE page before extracting data. Many websites load content dynamically, so you need to make sure you've seen everything before starting extraction.
        """

    try:
        llm = ChatOpenAI(model="gpt-4.1")
        agent = Agent(
            llm=llm,
            controller=controller,
            task=task,
            headless=False,
            file_system_path=Path.cwd(),
        )

        await agent.run()
    except Exception as e:
        print(f"Error running agent: {e}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract data from web pages using browser automation")
    parser.add_argument("entity_type", help="Type of data entity to extract (e.g., 'session/presentation', 'products', 'articles')")
    parser.add_argument("--url", "-u", help="Target URL to extract data from")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.entity_type, args.url))