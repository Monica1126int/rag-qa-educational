"""DeepSeek 生成模块（OpenAI 兼容接口）。"""
from openai import OpenAI

SYSTEM_PROMPT = (
    "你是一名高校教务知识库问答助手。请仅根据提供的参考资料回答用户问题；"
    "若资料中没有相关信息，请明确说明“根据现有资料无法回答”，不要编造。"
)


class Generator:
    """封装 DeepSeek 大模型生成调用。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, query: str, context: str) -> str:
        user = f"参考资料：\n{context}\n\n用户问题：{query}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
