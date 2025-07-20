from pydantic import BaseModel
from typing import List
import voyageai
import google.generativeai as genai
from google.generativeai import types
from bson import json_util
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()


# Configure Gemini API key
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
# Create Gemini model instance
llmClient = genai.GenerativeModel("gemini-2.0-flash")
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

    # Get full response using Gemini API
    response = llmClient.generate_content(prompt)
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
    import sys
    if len(sys.argv) > 1:
        userQuery = " ".join(sys.argv[1:])
    else:
        try:
            userQuery = input("query: ") # who is president of dhaked firm.
        except EOFError:
            print("No query provided. Exiting.")
            sys.exit(1)
    print("\n")
    queryVector = embedQueryVoyage(userQuery)
    dbData = queryMongoDB(queryVector)
    dataString = ".".join(item.data for item in dbData)
    print(f"RAG Data to be used for assisting generation: {dataString}\n")
    print(f"User Query: {userQuery}\n")
    answer = generateFinalResponseUsingGemini(dataString, userQuery)
    print(answer)

