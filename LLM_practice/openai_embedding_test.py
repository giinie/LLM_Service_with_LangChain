import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
document = ['제프리 힌튼', '교수', '토론토 대학', '사임']
# '제프리 힌튼', '교수', '토론토 대학', '사임'을 벡터로 변환
response = client.embeddings.create(
    input=document,
    # OpenAI 제공 임베딩 모델
    model="text-embedding-ada-002"
)
print(response)
