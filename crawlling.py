import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig, LLMContentFilter, DefaultMarkdownGenerator
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class NewsArticle(BaseModel):
    summary: str

# llm_config = LLMConfig(
#     provider="openai/gpt-4o",  # Specify your LLM provider
#     api_token=os.getenv("OPENAI_API_KEY")  # Provide your API token
# )

llm_config = LLMConfig(
    provider="gemini/gemini-2.0-flash",  # Specify your LLM provider
    api_token=os.getenv("GENAI_API_KEY"), # Provide your API token
    # base_url="https://generativelanguage.googleapis.com",
    # max_tokens=100000
)

llm_filter = LLMContentFilter(
    llm_config=llm_config,
    instruction="""
        You are an news agent.
        """,
    chunk_token_threshold=100000
)

llm_strategy = LLMExtractionStrategy(
    llm_config = llm_config,
    schema=NewsArticle.model_json_schema(), # Or use model_json_schema()
    extraction_type="schema",
    # instruction="Extract all product objects with 'name' and 'price' from the content.",
    instruction="You are an expert news agent. Extract the summary from the article.",
    chunk_token_threshold=1000,
    overlap_rate=0.0,
    apply_chunking=True,
    input_format="markdown",   # or "html", "fit_markdown"
    extra_args={"temperature": 0.0, "max_tokens": 800}
)


run_conf = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(content_filter=llm_filter,options={"ignore_links": True}),
    extraction_strategy=llm_strategy,
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
            data = json.loads(result.extracted_content) 
            print("Extracted content:", data)
            llm_strategy.show_usage()  # prints token usage
        else:
            print("Error:", result)
        print('\n')

if __name__ == "__main__":
    asyncio.run(main())
