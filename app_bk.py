import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),  # e.g., "gpt-4-0613" or "gpt-3.5-turbo"
)

response = llm.invoke("what is the capital of Vietnam?")
print(response.content)  # Output: "Hanoi"

# llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
#     model=os.getenv("AZURE_OPENAI_LLM_MODEL"),
#     api_version="2024-07-01-preview"
# )