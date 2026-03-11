from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)