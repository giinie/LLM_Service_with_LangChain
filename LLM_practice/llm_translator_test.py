import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 웹페이지에 보여질 내용
langs = ["Korean", "Japanese", "Chinese", "English"]  # 번영할 언어를 나열
left_co, cent_co, last_co = st.columns(3)

# 웹페이지 왼쪽에 언어를 선택할 수 있는 라디오 버튼
with st.sidebar:
    language = st.radio('번역을 원하는 언어를 선택해주세요:', langs)

st.markdown('### 언어 번역 서비스입니다~~~')
prompt = st.text_input('번역을 원하는 텍스트를 입력하세요')  # 사용자의 텍스트 입력

# 프롬프트를 번역으로 지시
trans_template = PromptTemplate(
    input_variables=['trans'],
    template='Your task is to translate this text to ' + language + 'TEXT: {trans}'
)

# memory는 텍스트 저장 용도
memory = ConversationBufferMemory(input_key='trans', memory_key='chat_history')

llm = ChatOpenAI(temperature=0.0, model='gpt-4')
trans_chain = LLMChain(llm=llm, prompt=trans_template, verbose=True, output_key='translate', memory=memory)

# 프롬프트(trans_template)가 있으면 이를 처리화고 화면에 응답을 작성
if st.button("번역"):
    if prompt:
        response = trans_chain({'trans': prompt})
        st.info(response['translate'])
