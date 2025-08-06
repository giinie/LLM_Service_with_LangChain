import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
# from langchain import FAISS
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

open_api_key = os.environ.get("OPENAI_API_KEY")


def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # 임베딩 처리(벡터 변환), 임베딩은 text-embedding-ada-002 모델을 사용
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=open_api_key)
    documents = FAISS.from_texts(chunks, embeddings)
    return documents


def main():  # streamlit을 이용한 웹사이트 생성
    st.title("📄PDF 요약하기")
    st.divider()

    pdf = st.file_uploader("PDF 파일을 업로드해주세요.", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""  # text 변수에 PDF 내용을 저장
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."  # LLM에 PDF파일 요약 요청

        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=open_api_key, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.invoke({"input_documents": docs, "question": query})
                print(cost)

            st.subheader('--요약 결과--:')
            st.write(response)


if __name__ == '__main__':
    main()
