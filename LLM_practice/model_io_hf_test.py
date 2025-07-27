import os

from langchain_huggingface import HuggingFaceEndpoint

llm_hf = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=os.environ.get("HF_GI_WORKS_TOKEN"),
    temperature=0.0,
    max_new_tokens=256,
)

prompt = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"
# completion = llm_hf(prompt)
completion = llm_hf.invoke(prompt)
print(completion)
