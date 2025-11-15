"""
A module to define a MessagePool class, which is used to record the historical conversation of an Agent.

"""

from llm_mas_ecrp.core.agent_team.base_agent import BaseAgent


class MessagePool:
    """
    A class to record the historical conversation of an Agent.

    Attributes:
    -----------
    sender : str
        The name or identifier of the sender of the message.
    receiver : str
        The name or identifier of the receiver of the message.
    content : str
        The content of the message.
    timestamp : datetime
        The time when the message was sent.

    Methods:
    --------
    __init__(self, sender, receiver, content, timestamp):
        Initializes a new instance of the Message class with the given sender, receiver, content, and timestamp.
    """

    def __init__(self):
        self.history_message = []  # 存储格式：{"agent": str, "content": str, "timestamp": datetime}

    def add_message(self, agent: BaseAgent, content: str, iter_: int):
        """添加对话记录"""
        self.history_message.append(
            {
                "agent": agent,
                "content": content,
                "timestamp": iter_,
            }
        )

    def get(self, agent: str) -> str:
        for msg in reversed(self.history_message):
            if msg["agent"].NAME == agent:
                return msg["content"]
        else:
            return ""

    def get_content(self) -> list[str]:
        """获取纯文本对话记录"""
        return [f"{msg['agent']}: {msg['content']}" for msg in self.history_message]

    def get_spoken_agents(self) -> list:
        """获取所有发言过的Agent对象"""
        return [msg["agent"] for msg in self.history_message]

    def pop(self) -> dict:
        return self.history_message.pop()

    def is_empty(self):
        return len(self.history_message) == 0

    def peek(self):
        if not self.is_empty():
            return self.history_message[-1]  # 返回栈顶元素
        else:
            raise IndexError("peek from empty stack")  # 栈为空时抛出异常
