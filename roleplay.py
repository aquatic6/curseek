import os
import json
import requests
import chromadb
from chromadb.config import Settings

# 初始化 Chroma 客户端
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
            timeout=30
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

def save_to_memory(user_input, ai_response, role):
    """保存交互到记忆"""
    try:
        full_interaction = f"用户: {user_input}\n{role}: {ai_response}"
        collection.add(
            documents=[full_interaction],
            ids=[f"id_{len(collection.get()['documents']) if collection.get() else 0}"]
        )
    except Exception as e:
        print(f"记忆保存错误: {e}")

def run_agent():
    print("角色扮演对话系统启动啦！输入'quit'退出程序。")
    
    # 获取用户想要AI扮演的角色
    print("\n请输入您希望我扮演的角色（例如：苏格拉底、心理咨询师、侦探等）：")
    role = input("> ").strip()
    
    if not role:
        role = "AI助手"
        print("未指定角色，使用默认角色: AI助手")
    else:
        print(f"好的，我将扮演{role}的角色与您对话。")
    
    # 构建角色特定的系统提示词
    system_prompt_template = f"""
你是一个专业的角色扮演AI。你必须完全沉浸在你所扮演的角色中，以{role}的身份、口吻和思维方式与用户进行对话。

## 角色设定
你正在扮演的角色是：{role}
- 思考方式：完全按照{role}的典型思维方式进行思考
- 语言风格：使用{role}可能会使用的语言风格和词汇
- 专业知识：展现{role}可能具备的专业知识和视角
- 互动方式：以{role}的身份与用户互动，保持角色一致性

## 对话规则
1. 始终保持角色一致性，绝不跳出角色
2. 使用提问的方式引导用户深入思考，而不是直接给出答案
3. 根据用户的回应逐步深入，提出更有深度的问题
4. 适当运用角色的专业知识和独特视角
5. 保持对话自然流畅，避免生硬的转换话题

## 输出格式
你必须严格按照以下JSON格式输出你的响应：
{{
    "response": "你的提问内容"  // 以{role}的身份提出的问题
}}

现在，请以{role}的身份开始与用户对话。
"""

    conversation_history = []

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'quit':
            break

        # 检索相关记忆
        relevant_memory = get_relevant_memory(user_input)
        
        # 准备完整的提示词
        full_prompt = f"{system_prompt_template}\n\n相关记忆:\n{relevant_memory}\n\n当前对话:\n"
        
        # 添加历史对话
        for i, msg in enumerate(conversation_history[-6:]):
            full_prompt += f"{msg}\n"
        
        full_prompt += f"用户: {user_input}\n{role}: "
        
        print("正在思考...")
        
        # 调用模型
        response = call_ollama(full_prompt)
        
        # 尝试解析JSON响应
        try:
            ai_response_json = json.loads(response)
            response_text = ai_response_json.get("response", "抱歉，我好像出错了。")
        except json.JSONDecodeError:
            response_text = response

        # 输出响应
        print(f"{role}: {response_text}")

        # 保存到记忆
        save_to_memory(user_input, response_text, role)
        
        # 更新会话历史
        conversation_history.append(f"用户: {user_input}")
        conversation_history.append(f"{role}: {response_text}")
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

    print("对话结束。")

if __name__ == "__main__":
    run_agent()