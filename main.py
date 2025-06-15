from pydantic import BaseModel
from typing import List
import voyageai
from bson import json_util
from google import genai
from google.genai import types
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os


# llm client
llmClient = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
# ollamaClient = Client(
#     host="localhost:11434"
# )


# connect to your Atlas cluster
uri = os.getenv("MONGO_DB_URI")  # e.g. "mongodb+srv://<username>:<password>@cluster.mongodb.net/test"
mongoClient = MongoClient(uri, server_api=ServerApi('1'))
mongoCollection = mongoClient["llm-vec-embeding-db"]["embeddings"]


# To generate query embedings
# embedingModelName = "mxbai-embed-large"   # mxbai-embed-large (ollama)
embedingModelName = "voyage-3.5"   # voyageai
vo = voyageai.Client(api_key=os.getenv("VOYAGEAI_API_KEY"))


# def embedQueryOllama(query:str) :
#   embedingResp = ollamaClient.embeddings(model = embedingModelName, prompt=query)
#   return embedingResp.embedding


def embedQueryVoyage(query:str) :
  data = [query]
  result = vo.embed(data, model=embedingModelName)
  vector = result.embeddings[0]
  return vector



def queryMongoDB(vector:List[float]):
  # define pipeline 
  pipeline = [
    {
      '$vectorSearch': {
          'index': 'llm-vec-embeding', 
          'path': 'data_embeded', 
          'queryVector': vector,
          'numCandidates': 200,
          'limit': 2
      }
    }, {
      '$project': {
        '_id': 0, 
        'data': 1, 
        'embeding_modal': 1,
        'score': {
          '$meta': 'vectorSearchScore'
        }
      }
    }
  ]
  resultSet = mongoCollection.aggregate(pipeline)
  dbResult: list[BaseEmbedingEntityLLM] = [BaseEmbedingEntityLLM.model_validate(doc) for doc in resultSet]
  print(dbResult)
  return dbResult


class BaseEmbedingEntityLLM(BaseModel):
  data: str
  embeding_modal: str
  score: float


class AIResponse(BaseModel):
    answer: str


def generateFinalResponseUsingGemini(prom_data:str, user_query:str) -> AIResponse:
    # Prepare the prompt with expected output type
    prompt = (
        f"You are a helpful assistant. Based on the following data:\n\n"
        f"{prom_data}\n\n"
        f"Please answer the user query:\n\"{user_query}\"\n\n"
        f"Return the response **only** in the following JSON format:\n\n"
        f'{{\n  "answer": "your answer here"\n}}\n\n'
        f"Make sure the response is strictly valid JSON."
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

    # Get full response
    response = llmClient.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    full_text = response.text.strip()

    # Parse the JSON response safely
    try:
        parsed = AIResponse.model_validate(json.loads(full_text.strip('```json').strip('`')))
    except Exception as e:
        raise ValueError(f"Failed to parse response: {full_text}") from e

    return parsed


# def generateFinalResponseUsingOllama(promt_data:str, user_data:str) -> AIResponse:
#     prompt = (
#         f"You are a helpful assistant. Based on the following data:\n\n"
#         f"{promt_data}\n\n"
#         f"Please answer the user query:\n\"{user_data}\"\n\n"
#         f"Return the response **only** in the following JSON format:\n\n"
#         f'{{\n  "answer": "your answer here"\n}}\n\n'
#         f"Make sure the response is strictly valid JSON."
#     )
    
#     ans = ollamaClient.generate(model="llama3.1:8b", prompt=prompt)

#     full_text = ans.response

#     # Parse the JSON response safely
#     try:
#         parsed = AIResponse.model_validate(json.loads(full_text.strip('```json').strip('`')))
#     except Exception as e:
#         raise ValueError(f"Failed to parse response: {full_text}") from e

#     return parsed



######################################################################################################################
if __name__ == "__main__":
  userQuery = input("query: ")    # "who is president id dhaked firm."
  print("\n")
  queryVector = embedQueryVoyage(userQuery)
  dbData = queryMongoDB(queryVector)
  
  dataString = ".".join(item.data for item in dbData)
  answer = generateFinalResponseUsingGemini(dataString,userQuery)
  print(answer)

