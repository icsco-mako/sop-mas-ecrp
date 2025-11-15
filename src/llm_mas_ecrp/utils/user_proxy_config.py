from openai import OpenAI
import time
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

client = {
    "DeepSeek": ["deepseek-v3", "deepseek-r1"],
    "OpenAI": ["gpt-4o", "o1", "o3-mini"],
    "Anthropic": ["claude-3.5-sonnet-20241022"],
    "Gemini": ["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp"],
}

__openai = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

__ark_client = OpenAI(
    base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),  # 替换为您需要调用的模型服务Base Url
    api_key=os.getenv("ARK_API_KEY"),
    timeout=1800,
)
# 字节DeepSeek: ep-20250217190713-6lfs8(deepseek-v3); ep-20250210181347-9n2pl(deepseek-r1);
# deepseek-v3-250324;deepseek-r1-250120


__tencent_client = OpenAI(
    api_key=os.getenv("TENCENT_API_KEY"),  # 如何获取API Key：https://cloud.tencent.com/document/product/1772/115970
    base_url=os.getenv("TENCENT_BASE_URL", "https://api.lkeap.cloud.tencent.com/v1"),
)  # tencent DeepSeek: deepseek-r1, deepseek-v3-0324

__anthropic = OpenAI(  # claude
    # 替换为您需要调用的模型服务Base Url
    base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.openai-proxy.org/v1"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

__gemini = OpenAI(
    # 替换为您需要调用的模型服务Base Url
    base_url=os.getenv("GEMINI_BASE_URL", "https://api.openai-proxy.org/v1"),
    api_key=os.getenv("GEMINI_API_KEY"),
)

__qwen = OpenAI(
    # 阿里云百炼
    # https://bailian.console.aliyun.com/#/model-market/detail/qwq-plus-2025-03-05?tabKey=sdk
    base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("QWEN_API_KEY"),
)  # model: qwq-32b

__deepseek = OpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


def get_llm_client(client: str) -> OpenAI:
    match client:
        case "OpenAI":
            return __openai
        case "DeepSeek":
            return __ark_client
        case "Anthropic":
            return __anthropic
        case "Gemini":
            return __gemini
        case "Qwen":
            return __qwen
        case _:
            raise ValueError("Invalid model name")


def timer(func):
    def inner(*args, **kwargs):
        start = time.time()
        gbest = func(*args, **kwargs)
        end = time.time()
        print(f"程序运行时间：{round(end - start, 2)}s")
        return gbest

    return inner


@timer
def test_tencent_client():
    completion = __tencent_client.chat.completions.create(
        model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=[
            {"role": "system", "content": "你是deepseek, AI 人工智能助手"},
            {"role": "user", "content": "常见的十字花科植物有哪些？"},
        ],
    )
    # 通过reasoning_content字段打印思考过程
    print("思考过程：")
    print(completion.choices[0].message.reasoning_content)
    # 通过content字段打印最终答案
    print("最终答案：")
    print(completion.choices[0].message.content)


@timer
def test_ark_client():
    print("----- standard request -----")
    completion = __ark_client.chat.completions.create(
        model="ep-20250210181347-9n2pl",
        messages=[
            {"role": "system", "content": "你是deepseek, AI 人工智能助手"},
            {"role": "user", "content": "常见的十字花科植物有哪些？"},
        ],
    )
    print(completion.choices[0].message.content)


def test_qwen():

    completion = __qwen.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen3-235b-a22b",  # model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
        stream=True,
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        extra_body={"enable_thinking": True},
    )
    response_text = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            response_text += content
            print(content, end="")  # 实时打印输出
    print("\n完整回复:", response_text)


def test_gemini_client():
    print("----- standard request -----")
    completion = __gemini.chat.completions.create(
        model="gemini-2.5-pro-preview-05-06",
        messages=[
            {"role": "system", "content": "你是AI 人工智能助手"},
            {"role": "user", "content": "常见的十字花科植物有哪些？"},
        ],
    )
    print(completion.choices[0].message.content)


def test_openai_client():
    print("----- standard request -----")
    completion = __openai.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=[
            {"role": "system", "content": "你是AI 人工智能助手"},
            {"role": "user", "content": "常见的十字花科植物有哪些？"},
        ],
    )
    print(completion.choices[0].message.content)


def test_deepseek():
    print("----- standard request -----")
    completion = __deepseek.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="deepseek-v3-1-250821",
        messages=[
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": "你好"},
        ],
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    # test_tencent_client()
    # test_ark_client()
    # test_qwen()
    # test_gemini_client()
    # test_openai_client()
    test_deepseek()
