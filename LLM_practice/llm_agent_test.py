from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
)

# Deprecated langchain.agents import load_tools
from langchain.agents import load_tools, initialize_agent, AgentType
# from langchain.agents import initialize_agent, AgentType
# from langchain_community.agent_toolkits import load_tools

tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, description="계산이 필요할 때 사용")
agent.invoke(input="에드 시런이 태어난 해는? 2025년도 현재 에드 시런은 몇 살?")