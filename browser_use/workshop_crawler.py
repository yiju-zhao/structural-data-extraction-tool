"""
Optimized Workshop Crawler for COLM 2025

A robust browser automation tool for extracting workshop information from conference websites.

Key Features:
- Comprehensive error handling with automatic retry mechanisms
- Structured logging with file and console output
- Configuration management via environment variables
- Graceful shutdown handling for SIGINT/SIGTERM
- Async file operations for better performance
- Proper validation and data sanitization
- Timestamped output files to prevent overwrites
- Respectful crawling with built-in delays

Usage:
    python workshop_crawler.py

Environment Variables:
    CRAWLER_MODEL: OpenAI model to use (default: gpt-5)
    CRAWLER_TEMP: Temperature setting (default: 0.7)
    CRAWLER_RETRIES: Max retry attempts (default: 3)
    CRAWLER_URL: Target URL to crawl
"""

import asyncio
import sys
import os
import pathlib
import logging
import signal
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import aiofiles

load_dotenv()

from browser_use import Agent, Controller
from browser_use.llm import ChatOpenAI

SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
agent_dir = SCRIPT_DIR / 'output'
agent_dir.mkdir(exist_ok=True)
conversation_dir = agent_dir / 'conversations'
conversation_dir.mkdir(exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utility.json_to_csv_converter import convert_browser_use_output_to_csv
except ImportError as e:
    logging.error(f"Failed to import converter utility: {e}")
    convert_browser_use_output_to_csv = None


class CrawlerConfig(BaseModel):
    target_url: str = "https://colmweb.org/workshops.html"
    model_name: str = "gpt-5"
    temperature: float = 1.0
    max_retries: int = 2
    retry_delay: int = 5
    output_dir: pathlib.Path = agent_dir
    save_raw_result: bool = True
    convert_to_csv: bool = True
    preserve_all_data: bool = True

    class Config:
        arbitrary_types_allowed = True


class Session(BaseModel):
    Title: str = Field(..., description="Workshop title")
    Abstract: str = Field(..., description="Workshop description/abstract")
    Speakers: str = Field(..., description="Invited speakers names and organizations")

    @field_validator('Title', 'Abstract', 'Speakers')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            return "N/A"
        return v.strip()


class Sessions(BaseModel):
    sessions: List[Session] = Field(..., description="List of workshop sessions")


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    log_dir = agent_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"workshop_crawler_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class WorkshopCrawler:
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.logger = setup_logging()
        self.controller = None
        self.agent = None
        self._shutdown_requested = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, _frame):
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

    async def _create_agent(self) -> Agent:
        try:
            self.controller = Controller(output_model=Sessions)
            
            task = f"""Extract structured information about each workshop listed on {self.config.target_url}. 

            Navigate through the workshop page and identify ALL workshops.
            For each workshop, visit the workshop website and extract the following information:

            1. Title: Title of the workshop.
            2. Abstract: It is normally the description paragraph of the workshop on the home page, use the original text don't modify it.
            3. Speakers: All the INVITED SPEAKERS' names (organization).

            Note: 
            1. Only include invited speakers—do not list program chairs, organizing committee, etc., unless they are clearly workshop invited speakers.
            2. If any data points are missing, mark them as "N/A" rather than leaving them blank.
            3. Be concise and extract only the necessary structured information. Skip workshops if their external website is broken or lacks desired information.
            4. Be respectful with your crawling - add delays between requests to avoid overwhelming servers."""

            self.agent = Agent(
                task=task,
                llm=ChatOpenAI(model=self.config.model_name, temperature=self.config.temperature),
                controller=self.controller,
                file_system_path=str(self.config.output_dir),
            )
            
            self.logger.info(f"Agent created with model {self.config.model_name}")
            return self.agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            raise

    async def _save_raw_result(self, result: Any) -> Optional[pathlib.Path]:
        if not self.config.save_raw_result or not result:
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file = self.config.output_dir / f"colm_workshop_raw_result_{timestamp}.txt"
            
            result_str = str(result)
            
            async with aiofiles.open(raw_file, 'w', encoding='utf-8') as f:
                await f.write(result_str)
                
            self.logger.info(f"Raw result saved to {raw_file} ({len(result_str)} characters)")
            return raw_file
            
        except Exception as e:
            self.logger.error(f"Failed to save raw result: {e}")
            return None

    async def _convert_to_csv(self, result: Any) -> Optional[pathlib.Path]:
        if not self.config.convert_to_csv or not convert_browser_use_output_to_csv:
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.config.output_dir / f"colm_workshop_results_{timestamp}.csv"
            
            result_str = str(result)
            
            stats = convert_browser_use_output_to_csv(
                raw_result=result_str,
                pydantic_model=Sessions,
                csv_filename=str(csv_file),
                preserve_all_data=self.config.preserve_all_data
            )
            
            self.logger.info(f"CSV conversion completed: {csv_file}")
            if stats:
                self.logger.info(f"Conversion stats: {stats}")
            return csv_file
            
        except Exception as e:
            self.logger.error(f"Failed to convert to CSV: {e}")
            return None

    async def run_with_retry(self) -> Optional[Any]:
        for attempt in range(1, self.config.max_retries + 1):
            if self._shutdown_requested:
                self.logger.info("Shutdown requested, aborting crawl")
                return None
                
            try:
                self.logger.info(f"Attempt {attempt}/{self.config.max_retries}: Starting crawl")
                
                agent = await self._create_agent()
                history = await agent.run()
                result = history.final_result()
                
                if result:
                    self.logger.info("Crawl completed successfully")
                    return result
                else:
                    self.logger.warning(f"Attempt {attempt}: No result returned")
                    
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {e}")
                
                if attempt < self.config.max_retries:
                    self.logger.info(f"Retrying in {self.config.retry_delay} seconds...")
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    self.logger.error("All retry attempts exhausted")
                    
        return None

    async def run(self) -> bool:
        self.logger.info("Starting workshop crawler")
        
        try:
            result = await self.run_with_retry()
            
            if not result:
                self.logger.error("Crawler failed to extract data")
                return False
                
            self.logger.info("Processing results...")
            
            raw_file = await self._save_raw_result(result)
            csv_file = await self._convert_to_csv(result)
            
            self.logger.info("Crawler completed successfully")
            if raw_file:
                self.logger.info(f"Raw result: {raw_file}")
            if csv_file:
                self.logger.info(f"CSV result: {csv_file}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Crawler failed with error: {e}")
            return False


async def main():
    try:
        config = CrawlerConfig()
        
        # Allow configuration via environment variables
        if os.getenv('CRAWLER_MODEL'):
            config.model_name = os.getenv('CRAWLER_MODEL')
        if os.getenv('CRAWLER_TEMP'):
            config.temperature = float(os.getenv('CRAWLER_TEMP'))
        if os.getenv('CRAWLER_RETRIES'):
            config.max_retries = int(os.getenv('CRAWLER_RETRIES'))
        if os.getenv('CRAWLER_URL'):
            config.target_url = os.getenv('CRAWLER_URL')
            
        crawler = WorkshopCrawler(config)
        success = await crawler.run()
        
        if success:
            print("\n✅ Workshop crawler completed successfully!")
            return 0
        else:
            print("\n❌ Workshop crawler failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Crawler interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
