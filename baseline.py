import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class Baseline:
    
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        
        self.llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model_name,
            temperature=0.8,
            streaming=False,
            max_tokens=8192
        )
    
    def generate(self, user_input: str) -> Dict[str, Any]:
        start_time = time.time()
        
        system_prompt = """你是一个专业的剧本编剧。请根据用户需求生成一个完整的剧本。
                        剧本必须包含以下部分，
                        1.【剧情简介】100字左右
                        2.【角色列表】每个角色的姓名、年龄、性格
                        3.【场景与对话】至少3个场景，每个场景5-10轮对话
                        4.【分镜脚本】为每个场景设计分镜头
                        请确保剧本完整、逻辑连贯、对话自然。"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        response_text = ""
        try:
            response = self.llm.invoke(messages)
            response_text = response.content
        except Exception as e:
            response_text = f"生成失败: {e}"
        
        gen_time = time.time() - start_time
        
        return {
            "input": user_input,
            "output": response_text,
            "metadata": {
                "generation_time": gen_time,
                "model": self.model_name,
                "method": "baseline"
            }
        }


if __name__ == "__main__":
    llm_client = Baseline(model_name="qwen-plus-2025-01-25")
    
    while True:
        user_input = input("\n请输入剧本需求：").strip()
        
        if not user_input:
            print("输入不能为空，请重新输入")
            continue
        
        print("\n正在生成剧本，请稍候...")
        
        result = llm_client.generate(user_input)
        
        print(f"生成完成，耗时: {result['metadata']['generation_time']:.2f}秒")
        print(f"\n{result['output']}")