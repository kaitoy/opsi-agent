import asyncio

import nest_asyncio
import uvicorn
from fastapi import FastAPI, status
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.pydantic_v1 import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langserve import add_routes
from pydantic import BaseModel as BaseModelV2

nest_asyncio.apply()

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 400,
    chunk_overlap = 0,
    length_function = len,
)

loader = SitemapLoader(web_path="https://itpfdoc.hitachi.co.jp/manuals/JCS/JCSM71020002/sitemap.xml")

vic = VectorstoreIndexCreator(
    vectorstore_cls=PGVector,
    embedding=OpenAIEmbeddings(),
    text_splitter=text_splitter,
    vectorstore_kwargs={
        "collection_name": "opsi_manual",
        "connection": "postgresql+psycopg://langchain:langchain@opsi-agent-postgres:5432/langchain",
        "use_jsonb": True,
    }
)
index = asyncio.run(vic.afrom_loaders([loader]))

retriever = index.vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """contextに基づいて、Ops Iの質問になるべく頑張って答えてください。ただし、Ops Iと関係ない質問に対しては、知るかボケと回答してもいいです:

<context>
{context}
</context>
"""
    ),
    (
        "human",
        "質問: {input}"
    )
])
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

class ChainInput(BaseModel):
    input: str
app = FastAPI(
    title="Ops I Assistant",
    version="1.0",
    description="Ops I Assistant",
)
add_routes(
    app,
    chain.with_types(input_type=ChainInput),
    path="/opsi",
)

class HealthCheck(BaseModelV2):
    status: str = "OK"
@app.get(
    "/healthz",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")

uvicorn.run(app, host="0.0.0.0", port=8080)
