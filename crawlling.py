import json
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, DefaultMarkdownGenerator, CacheMode, LLMConfig, LLMContentFilter
from pydantic import BaseModel
import asyncio
import os


class NewsArticle(BaseModel):
    tile: str
    description: str


# llm_config = LLMConfig(
#     provider="openai/gpt-4o",  # Specify your LLM provider
#     api_token=os.getenv("OPENAI_API_KEY")  # Provide your API token
# )

llm_config = LLMConfig(
    provider="gemini/gemini-2.0-flash",  # Specify your LLM provider
    api_token=os.getenv("GENAI_API_KEY"), # Provide your API token
    base_url="https://generativelanguage.googleapis.com",
    max_tokens=100000
)

llm_strategy = LLMContentFilter(
    llm_config=llm_config,
    instruction="""
        You are an news agent.
        """,
    chunk_token_threshold=1000
)

run_conf = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(content_filter=filter,options={"ignore_links": True}),
    cache_mode=CacheMode.BYPASS,
)

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://timesofindia.indiatimes.com/india/chhattisgarh-cops-not-keen-to-give-basavarajus-body-to-family/articleshow/121398834.cms",
            config=run_conf
        )
        print('\n')
        
        if result.success:
            print("Extracted content:", result)
        else:
            print("Error:", result)
        print('\n')

if __name__ == "__main__":
    asyncio.run(main())
