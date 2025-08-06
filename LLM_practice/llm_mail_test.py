import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="이메일 작성 서비스",
    page_icon=":robot:",
)
st.header("이메일 작성기")


def get_email():
    input_mail_text = st.text_area(
        label="메일 입력",
        label_visibility='collapsed',
        placeholder="당신의 메일은...",
        key="input_text",
    )
    return input_mail_text


input_text = get_email()

# 이메일 변환 작업을 위한 템플릿 정의
query_template = """
    메일을 작성해주세요.
    아래는 이메일입니다.
    이메일: {email}
"""

# PromptTemplate 인스턴스 생성
prompt = PromptTemplate(
    input_variables=["email"],
    template=query_template,
)


# 언어 모델을 호출합니다.
def load_language_model():
    lang_model = ChatOpenAI(temperature=0.0, model='gpt-4')
    return lang_model


# 예시 이메일 표시
st.button(
    "*예제를 보여주세요*",
    type='secondary',
    help="봇이 작성한 메일을 확인해보세요."
)
st.markdown("### 봇이 작성한 메일은:")

if input_text:
    llm = load_language_model()
    # PromptTemplate 및 언어 모델을 사용하여 이메일 형식 지정
    prompt_with_email = prompt.format(email=input_text)
    formatted_email = llm.invoke(prompt_with_email)
    # 서식이 지정된 이메일 표시
    st.write(formatted_email)
