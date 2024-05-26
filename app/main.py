import nest_asyncio
import uvicorn
from fastapi import FastAPI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes


nest_asyncio.apply()

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 400,
    chunk_overlap = 0,
    length_function = len,
)

loader = SitemapLoader(web_path="https://itpfdoc.hitachi.co.jp/manuals/JCS/JCSM71020002/sitemap.xml")

index = VectorstoreIndexCreator(
    vectorstore_cls=InMemoryVectorStore,
    embedding=OpenAIEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])

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

app = FastAPI(
    title="Ops I Assistant",
    version="1.0",
    description="Ops I Assistant",
)
add_routes(
    app,
    chain,
    path="/opsi",
)

uvicorn.run(app, host="localhost", port=8080)
