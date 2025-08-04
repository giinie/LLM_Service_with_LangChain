import os

from langchain.chains.question_answering import load_qa_chain
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

documents = TextLoader("../Data/AI.txt").load()


# 문서를 청크로 분할
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# docs 변수에 분할 문서를 저장
docs = split_docs(documents)

open_api_key = os.environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=open_api_key)

# ChromaDB에 벡터 저장
db = Chroma.from_documents(docs, embeddings)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=open_api_key,
)

# Q&A 체인을 사용하여 쿼리에 대한 답변 얻기
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# 쿼리를 작성하고 유사도 검색을 수행하여 답변을 생성, 따라서 텍스트에 있는 내용을 질의해야 합니다.
query = "AI란?"
matching_docs = db.similarity_search(query)
answer = chain({"input_documents": matching_docs, "question": query})
print(answer)
