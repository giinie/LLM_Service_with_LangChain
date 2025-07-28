# from langchain.chains import LLMChain # Deprecated since 0.1.17
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(
    temperature=0,
    model="gpt-4o-mini",
)

prompt = PromptTemplate(
    input_variables=["country"],
    template="{country}의 수도는 어디야?",
)

# chain = LLMChain(llm=llm, prompt=prompt) # 프롬프트와 모델을 체인으로 연결 # Deprecated since 0.1.17
chain = prompt | llm | StrOutputParser()

chain.invoke("대한민국")