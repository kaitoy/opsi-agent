import sys

from langchain_openai import ChatOpenAI

input = sys.argv[1]

llm = ChatOpenAI(model="gpt-3.5-turbo")

response = llm.invoke(input)
print(response)
