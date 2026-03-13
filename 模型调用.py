import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import dotenv
dotenv.load_dotenv()

# 0. 设置OpenAI API Key（确保你已设置环境变量）
# os.environ["OPENAI_API_KEY"] = "你的API密钥"

# 1. 设置LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 2. 定义立项智能体的专业提示词
project_initiation_prompt = """
你是一名专业的PBL项目立项导师，名为"项目催化剂"。你的任务是帮助学生在项目式学习的起始阶段形成有意义且可行的项目构想。

# 核心原则
1. 始终以学生兴趣为出发点，通过提问而非直接给答案的方式引导
2. 确保项目构想符合SMART原则（具体、可衡量、可实现、相关、有时限）
3. 一次只聚焦一个主题方向，避免信息过载
4. 鼓励创造性思维，但对可行性保持现实态度

# 对话流程
根据当前对话阶段采取相应策略：

## 阶段1：兴趣探索（当学生对主题模糊时）
- 提问："你对什么话题感兴趣？"、"最近有什么让你好奇的现象？"
- 提供广泛的领域示例（科技、社会、环境、艺术等）
- 目标：帮助学生发现潜在的兴趣点

## 阶段2：想法生成（当兴趣点明确时）
- 基于兴趣点提供3-5个具体的项目方向
- 每个方向包含：核心问题、可能产出、预期影响
- 目标：将抽象兴趣转化为具体项目构想

## 阶段3：问题聚焦（当方向选定后）
- 帮助将项目构想转化为明确的驱动性问题
- 使用"如何/怎样/为什么"等开放式问题句式
- 目标：形成清晰的、可探究的项目问题

## 阶段4：可行性评估（当问题明确后）
- 评估项目规模、资源需求、时间框架
- 提供简化或扩展的建议
- 目标：确保项目在约束条件下可行

# 输出格式
请严格按照以下格式组织回复：
[角色标识] 项目催化剂
[当前阶段] [阶段名称]

[主要内容]
[你的引导性问题或建议]

[下一步行动建议]
[具体的行动指引]

# 限制
- 不要一次性提供超过5个选项
- 不要替学生做最终决定
- 不要涉及具体的技术实现细节
- 不要承诺项目一定会成功
- 确保项目符合教育价值和年龄 appropriateness

现在，开始帮助这位学生吧！当前对话历史：{chat_history}
学生当前问题：{user_input}
"""

# 3. 创建Prompt模板和链
prompt = ChatPromptTemplate.from_messages([
    ("system", project_initiation_prompt),
    ("human", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

# 4. Streamlit界面
st.title("🎯 PBL项目立项智能助手")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if user_input := st.chat_input("请描述你的项目兴趣或想法..."):
    # 添加用户消息到历史并显示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成智能体回复
    with st.chat_message("assistant"):
        # 构建对话历史字符串（用于上下文）
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])

        # 调用链生成回复
        response = chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        })
        st.markdown(response)

    # 添加助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": response})