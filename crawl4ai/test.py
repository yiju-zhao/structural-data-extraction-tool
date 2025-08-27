from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

import asyncio

md_generator = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
)

config = CrawlerRunConfig(
    # cache_mode=CacheMode.BYPASS,
    markdown_generator=md_generator
)

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://news.ycombinator.com", config=config)
        print("Raw Markdown length:", len(result.markdown.raw_markdown))
        print("Fit Markdown length:", len(result.markdown.fit_markdown))

if __name__ == "__main__":
    asyncio.run(main())