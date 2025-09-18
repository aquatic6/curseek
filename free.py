import os
import json
import requests
import chromadb
from chromadb.config import Settings

# 系统提示词模板
system_prompt_template = """

角色与模式上下文

你是一个专业的“引导式提问”专家。你的唯一目标是：通过精心设计的问题序列，引导用户厘清思路、深入思考、自主发现答案或解决方案。无论用户提出何种问题或需求，你都必须遵循以下严格规则。
严格规则

    明确目标与背景：首先，你需要了解用户提问的真正目的和背景。如果用户的需求模糊，你的第一个问题必须是轻量级的，旨在澄清目标（例如：“你能多告诉我一些吗？你希望最终达成什么？”）。如果用户未提供详细信息，则默认以帮助其进行逻辑分析和深度思考为目标。

    建立于已知信息之上：从用户已提供的信息或普遍认知出发，将复杂问题分解，并将新问题与用户已有的陈述联系起来。

    引导，而非给予：绝对不要直接给出答案、解决方案或最终结论。你的任务是使用提问、提示、反问和渐进式步骤，让用户自己一步步构建出答案。每个回复只提出一个核心问题或一个最小的后续步骤，并必须等待用户回应后再继续。

    确认与强化理解：在完成一个关键的思考环节后，通过提问确认用户的理解（例如：“所以，根据我们刚才说的，你觉得下一步的关键是什么？”）。在适当时机，提供简短的总结，帮助用户巩固思维路径。

    保持对话节奏：混合使用不同类型的提问（如开放式问题、封闭式确认、假设性提问），让对话自然流畅。根据用户的回应，灵活决定是深入挖掘某个点，还是转向下一个环节。避免冗长的独白。

最重要的是：绝对不要替用户完成他们自己的思考工作。 不要直接解决问题，而是通过与用户协作，从他们已知的信息出发，引导他们自己找到答案。
你可以执行的任务

    澄清模糊需求：当用户需求不明确时，通过提问帮助其聚焦核心问题。

    分解复杂问题：将一个大问题拆解成多个可操作的小问题，逐步引导用户分析。

    激发创造性思维：通过假设性提问（“如果……会怎样？”）、反向提问（“为什么它不会发生？”）等方式，拓宽用户的思考维度。

    决策辅助：通过列出利弊、评估标准、潜在风险等提问，引导用户自行做出权衡，而非直接建议选择。

    复盘与总结：在对话后期，引导用户自己总结讨论出的要点和结论，确保思考的闭环。

语气与风格

保持温暖、中立且专业的语气。避免使用过多感叹号或表情符号。你的提问应清晰、简洁、一针见血。始终掌控对话的节奏，知道每一步的目标，并在一个环节完成后流畅地过渡到下一个。你的回复长度应适中，以推动对话高效进行。
重要禁令

绝对禁止代替用户思考或直接提供答案。 即使用户上传了图片、文档或直接提出了一个具体问题，你的第一反应也绝不能是解决问题本身。相反，你必须：

    根据用户提供的材料，提出第一个引导性问题。

    等待用户的回答。

    根据用户的回答，提出下一个问题。

    如此循环，直至用户在自己的思考中找到答案。

## 输出格式
你必须严格按照以下格式输出你的响应，绝不要输出任何其他内容：
{{
    "mode": "answer" | "clarify", // 选择模式
    "response": "你的回答内容" | "你的提问内容" // 根据模式填写
}}

现在，请结合之前的对话历史和相关记忆，来回应用户的最新输入。
"""

# 初始化 Chroma 客户端 - 使用新版本的 API
client = chromadb.PersistentClient(path="./chroma_db")

# 创建或获取集合
collection = client.get_or_create_collection(name="chat_history")

def call_ollama(prompt):
    """调用 Ollama API"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-coder-v2",
                "prompt": prompt,
                "temperature": 0.7,
                "stream": False
            },
            timeout=2000  # 添加超时设置
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"API调用失败: {str(e)}"

def get_relevant_memory(query, n_results=3):
    """从向量数据库获取相关记忆"""
    try:
        # 使用新的查询API
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if results and 'documents' in results and results['documents']:
            return "\n".join(results['documents'][0])
        return "暂无相关记忆。"
    except Exception as e:
        print(f"记忆检索错误: {e}")
        return "暂无相关记忆。"

def save_to_memory(user_input, ai_response, mode):
    """保存交互到记忆"""
    try:
        full_interaction = f"用户: {user_input}\nAI: {ai_response} (模式: {mode})"
        # 使用新的添加API
        collection.add(
            documents=[full_interaction],
            ids=[f"id_{len(collection.get()['documents']) if collection.get() else 0}"]
        )
    except Exception as e:
        print(f"记忆保存错误: {e}")

def run_agent():
    print("好问Curseek启动啦！输入'quit'退出程序。")
    conversation_history = []

    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break

        # 检索相关记忆
        relevant_memory = get_relevant_memory(user_input)
        
        # 准备完整的提示词
        full_prompt = f"{system_prompt_template}\n\n相关记忆:\n{relevant_memory}\n\n当前对话:\n"
        
        # 添加历史对话
        for i, msg in enumerate(conversation_history[-6:]):  # 只保留最近6条历史
            full_prompt += f"{msg}\n"
        
        full_prompt += f"用户: {user_input}\nAI: "
        
        print("正在思考...")  # 添加提示，因为本地模型可能需要一些时间
        
        # 调用模型
        response = call_ollama(full_prompt)
        
        # 尝试解析JSON响应
        try:
            ai_response_json = json.loads(response)
            mode = ai_response_json.get("mode", "answer")
            response_text = ai_response_json.get("response", "抱歉，我好像出错了。")
        except json.JSONDecodeError:
            mode = "answer"
            response_text = response

        # 根据模式输出
        if mode == "answer":
            print(f"好问(回答模式): {response_text}")
        else:
            print(f"好问(提问模式): {response_text}")

        # 保存到记忆
        save_to_memory(user_input, response_text, mode)
        
        # 更新会话历史（限制历史长度）
        conversation_history.append(f"用户: {user_input}")
        conversation_history.append(f"AI: {response_text}")
        if len(conversation_history) > 10:  # 最多保留10条历史
            conversation_history = conversation_history[-10:]

    print("对话结束。")

if __name__ == "__main__":
    run_agent()