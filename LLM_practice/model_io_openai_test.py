# from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

llm_openai = ChatOpenAI(
    temperature=0, # 창의성 0으로 설정
    max_tokens=2048, # 최대 토큰 수
    model_name='gpt-4', # 모델 명
)

output_parser = CommaSeparatedListOutputParser() # 파서 초기화
format_instructions = output_parser.get_format_instructions() # 출력 형식 지정

prompt_template = PromptTemplate(
    template="7개의 팅을 보여줘 {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

prompt_openai = "한국의 야구팀은?"
# print(llm1.predict(prompt_openai)) # Deprecated
response = llm_openai.invoke(input=prompt_template.format(subject=prompt_openai))

# 출력에 대한 포맷 변경
parsed_result = output_parser.parse(response.content)
print(parsed_result)