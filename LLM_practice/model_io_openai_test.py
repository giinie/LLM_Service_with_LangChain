# from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = "{product}를 홍보하기 위한 좋은 문구를 추천해줘"

prompt = PromptTemplate(input_variables=["product"], template=template)

prompt.format(product="카메라")

print(prompt)

llm1 = ChatOpenAI(
    temperature=0.0,
    model_name='gpt-4',
)

prompt_openai = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"
# print(llm1.predict(prompt_openai)) # Deprecated
response = llm1.invoke(prompt_openai)
print(response.content)