# 파이썬 언어로 작성된 데이터를 분석 및 조작하기 위한 라이브러리
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# csv 파일을 데이터프레임으로 가져오기
df = pd.read_csv('../Data/booksv_02.csv')
# print(df.head())

# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4o"),
    df,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
)

response = agent.invoke('어떤 제품의 ratings_count가 제일 높아?')
print(response)

response = agent.invoke('가장 최근에 출간된 책은?')
print(response)
