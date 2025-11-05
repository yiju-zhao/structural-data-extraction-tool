#!/usr/bin/env python3
"""
LLM-based Web Content Extractor using Crawl4AI
"""

import os
import json
from typing import List, Type, Union, Tuple
from pydantic import BaseModel
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
)
from crawl4ai import LLMExtractionStrategy

from dotenv import load_dotenv
load_dotenv()

class LLMExtractor:
    """
    A reusable LLM-based web content extractor using Crawl4AI.

    This class encapsulates the logic for extracting structured data from web pages
    using LLM (Large Language Model) with Pydantic schemas.
    """

    def __init__(
        self, model: str = "gpt-4o", headless: bool = True, verbose: bool = False
    ):
        """
        Initialize the LLM Extractor.

        Args:
            model: OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o")
            headless: Run browser in headless mode
            verbose: Print detailed logs
        """
        self.model = model
        self.api_token = os.getenv("OPENAI_API_KEY")
        self.headless = headless
        self.verbose = verbose

        if not self.api_token:
            raise ValueError("OPENAI_API_KEY environment variable is required.")

    async def extract(
        self,
        url: str,
        schema: Type[BaseModel],
        instruction: str,
        apply_chunking: bool = False,
        chunk_token_threshold: int = 4000,
        overlap_rate: float = 0.1,
        input_format: str = "markdown",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> Tuple[Union[BaseModel, List[BaseModel]], LLMExtractionStrategy]:
        """
        Extract structured data from a URL using LLM.

        Args:
            url: URL to crawl
            schema: Pydantic BaseModel class defining the extraction schema
            instruction: Extraction instruction for the LLM
            apply_chunking: Whether to apply chunking for large content
            chunk_token_threshold: Token threshold for chunking
            overlap_rate: Overlap rate for chunks
            input_format: Input format ("markdown", "html", "fit_markdown")
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM response

        Returns:
            Tuple of (extracted_data, llm_strategy)
            - extracted_data: Extracted content (BaseModel instance or list of instances)
            - llm_strategy: LLMExtractionStrategy instance with usage stats
        """

        # Define the LLM extraction strategy
        llm_strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider=f"openai/{self.model}", api_token=self.api_token
            ),
            schema=schema.model_json_schema(),
            extraction_type="schema",
            instruction=instruction,
            apply_chunking=apply_chunking,
            chunk_token_threshold=chunk_token_threshold,
            overlap_rate=overlap_rate,
            input_format=input_format,
            extra_args={"temperature": temperature, "max_tokens": max_tokens},
        )

        # Build the crawler config
        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy, cache_mode=CacheMode.BYPASS
        )

        # Create a browser config
        browser_cfg = BrowserConfig(headless=self.headless)

        # Run extraction
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            if self.verbose:
                print(f"  🔄 Crawling: {url}")

            result = await crawler.arun(url=url, config=crawl_config)

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)

                if self.verbose:
                    item_count = len(data) if isinstance(data, list) else 1
                    print(f"  ✅ Extracted {item_count} item(s)")

                # Convert to Pydantic model
                if isinstance(data, list):
                    extracted_data = [schema(**item) for item in data]
                else:
                    extracted_data = schema(**data)

                return extracted_data, llm_strategy
            else:
                if self.verbose:
                    print(f"  ❌ Extraction failed: {result.error_message}")

                # Return empty result
                return [], llm_strategy

    def show_usage(self, llm_strategy: LLMExtractionStrategy):
        """Show token usage statistics."""
        if llm_strategy:
            llm_strategy.show_usage()
