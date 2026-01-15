from Agents import User, Planner
from dotenv import load_dotenv
import os
from model import CustomLLM

from crewai.flow.flow import Flow, listen, start
from typing import Any, Dict, List
from pydantic import BaseModel, Field, PrivateAttr


load_dotenv()
# 从环境变量获取 API 密钥，如果不存在则使用默认值
api_key = os.environ.get("OPENAI_API_KEY", "bd4e0cd0cd0b49e4ca7ad1767baadf3a09cbab24f7aa6a9a8486cd7e3b9d9eaf")
model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
endpoint = os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions")
llm = CustomLLM(api_key=api_key, model=model_name, endpoint=endpoint, temperature=0.0, top_p=1.0)  

class MainFlowState(BaseModel):
    userInput: str = Field("", description="The user input for the flow")


class MainFlow(Flow[MainFlowState]):
    def __init__(self):
        super().__init__()
        
    @start()
    def UserInputProcess(self):
        userAgent = User(llm=llm)
        ProcessedUserInput = userAgent.kickoff(self.state.userInput)
        print(ProcessedUserInput.raw)
        return ProcessedUserInput.raw

    @listen(UserInputProcess)
    def PlannerProcess(self, ProcessedUserInput: str):
        planner = Planner(llm=llm)
        return planner.kickoff(ProcessedUserInput).raw
        

def main():
    state = {
        "userInput": "帮我看看最近一个月的攻击事件"
    }
    flow = MainFlow()
    flow.plot("mainflow.html")
    result = flow.kickoff(state)
    print(result)

if __name__ == "__main__":
    main()
