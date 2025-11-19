import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# -------------------------------
# 1) 상태 정의
# -------------------------------
class AgentState(TypedDict):
    question: str
    answer: str


# -------------------------------
# 2) LLM 노드 정의
# -------------------------------
def robotics_llm_node(state: AgentState) -> AgentState:
    """로보틱스 개발자 관점에서 답변하는 Node"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    prompt = f"""
당신은 고급 로보틱스 개발자입니다.
ROS2, SLAM, Nav2, Jetson, VLM/LLM, 센서퓨전, Localization, Control 등 
로봇 시스템 전체를 깊게 이해하고 있습니다.

아래 질문에 대해 '로보틱스 개발자의 시각'으로
전문적이고 구조적인 답변을 제공하세요.

질문:
{state["question"]}
"""

    result = llm.invoke(prompt)

    return {"answer": result.content}


# -------------------------------
# 3) LangGraph 그래프 구성
# -------------------------------
graph = StateGraph(AgentState)
graph.add_node("robotics_llm", robotics_llm_node)
graph.set_entry_point("robotics_llm")
graph.add_edge("robotics_llm", END)

app = graph.compile()


# -------------------------------
# 4) 실행 예제
# -------------------------------
if __name__ == "__main__":
    user_input = "Jetson Orin에서 Nav2를 안정적으로 돌리기 위한 핵심 튜닝 요소는?"
    result = app.invoke({"question": user_input})
    print("\n=== Agent 답변 ===\n")
    print(result["answer"])
