from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
)

loader = PyPDFLoader("../Data/The_Adventures_of_Tom_Sawyer.pdf")
document = loader.load()

# print(document[5].page_content[:5000])

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(document, embeddings)

retriever = vectorstore.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

query = "마을 무덤에 있던 남자를 죽인 사람은 누구니?"
# result = qa(query)
result = qa.invoke(query)
print(result["result"])
