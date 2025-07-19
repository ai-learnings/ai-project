# AI Weather Information System

This project aims to create an AI system that provides the latest weather information using advanced technologies:

- **LLM (Large Language Models) & GenAI Prompting**
- **RAG (Retrieval-Augmented Generation)**
- **Tools**: Crawling (e.g., `crawl4ai`), web searches
- **Agents**
- **MCPs**: MCP client & server

## Features

- Better Prompting
- MongoDB as Knowledgebase for RAG
- Use of tools for web searches (via `crawl4ai`)
- Agent-based architecture
- MCP client/server integration

---

## Environment Variables

Create a `.env` file in the project root and add your API keys and MongoDB URI:

```env
GENAI_API_KEY=<api-key>
OPENAI_API_KEY=<api-key>
MONGO_DB_URI=<connection-url>
VOYAGEAI_API_KEY=<api-key>
```

## MongoDB indexes
Search vector index:
```
{
  "fields": [
    {
      "type": "vector",
      "path": "data_embeded",
      "numDimensions": 1024,
      "similarity": "cosine"
    }
  ]
}
```
