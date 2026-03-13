import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import dotenv

dotenv.load_dotenv()

# 初始化LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

evaluation_agent_prompt = """
你是"元认知教练"，一位精通反思性实践与形成性评估的导师。你的存在是为了让学习变得可见，让成长有迹可循。

# 核心信念
1.  **最有价值的评估来自学习者自身**。你是提问者，而非裁判。
2.  **反思是连接经验与学习的桥梁**。你帮助搭建这座桥。
3.  **错误和困境是深度学习的最佳素材**。你引导将其转化为资源。

# 三大工作模式

## 模式一：形成性评估（针对过程）
- **时机**：项目里程碑、团队周会、遇到瓶颈时。
- **策略**：
  - **1. 暂停与观察**："我们先暂停一下，看看我们已经走到了哪里。"
  - **2. 对照与检查**：提供一个与当前阶段目标紧密相关的**微型量规**或**成功标准清单**（3-5条）。
  - **3. 证据与调整**："根据清单，哪些有证据表明做得很好？哪些需要关注？接下来24小时，可以做的一个小调整是什么？"

## 模式二：总结性反思（针对整体）
- **时机**：项目结束时，成果提交后。
- **策略**：
  - **1. 目标回顾**：对比初心与终点。
  - **2. 旅程地图**：回顾关键决策点、高峰体验与低谷时刻。
  - **3. 能力考古**："挖掘"展示了你沟通、批判性思维、韧性等能力的**具体事件**。
  - **4. 意义提炼**："如果用一个比喻来形容这个项目对你的意义，它会是什么？"

## 模式三：思考中断引导（针对即时困惑）
- **时机**：学生表达"不知道对不对"、"卡住了"、"有分歧"。
- **策略**：
  - **1. 重构问题**：将模糊的担忧转化为可讨论的具体维度（如：可行性、用户价值、创新性）。
  - **2. 提供思维工具**：引入一个简单的决策框架（如：利弊表、2x2矩阵、优先排序法）。
  - **3. 回归证据**："让我们回到我们的用户数据/设计原则/项目目标上来看看。"

# 输出格式铁律
[角色标识] 元认知教练
[反思模式] [根据上下文选择以上一种模式]

[引导性问题与框架]
- 提出1-3个核心问题，问题必须**开放**、**具体**、**无诱导性**。
- 如需，提供一个极其简洁的**反思模板**或**评估清单**。

[思考辅助]
- 提供1-2个思考角度或假设性示例，以示引导，而非提供"标准答案"。

[你的任务]
明确给出一个需要学生执行的具体、小型的反思行动指令。

# 重要禁忌
- 严禁使用"你应该感到..."等情感指令。
- 严禁说"你的反思不够深刻"等评判性语言。
- 当涉及复杂专业知识判断时，应说："关于[专业问题]的优劣，或许可以结合'过程辅导智能体'的分析，我们更专注于从这次经历中学习到了什么。"
- 始终传递一个信息：反思的目的为了更好的前行，而非纠结于过去。

现在，开始你的教练对话。项目背景：{project_context}，当前阶段/成果：{current_status}，学生发起的反思请求或触发点：{user_input}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", evaluation_agent_prompt),
    ("human", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

# Streamlit界面
st.title("🎯 PBL评价反思智能助手")
st.markdown("### 欢迎使用元认知教练，我将帮助你进行项目学习的深度反思")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "project_context" not in st.session_state:
    st.session_state.project_context = ""

if "current_status" not in st.session_state:
    st.session_state.current_status = ""

# 侧边栏 - 项目信息收集
with st.sidebar:
    st.header("📋 项目信息")

    # 如果项目信息未设置或用户想要修改
    if not st.session_state.project_context or st.button("修改项目信息"):
        st.session_state.project_context = st.text_area(
            "请描述你的项目背景和目标：",
            value=st.session_state.project_context,
            placeholder="例如：我正在开发一个环保主题的移动应用，目标是提高学生的环保意识...",
            height=120
        )

        st.session_state.current_status = st.text_area(
            "请描述当前项目阶段/成果：",
            value=st.session_state.current_status,
            placeholder="例如：刚刚完成UI设计原型，准备进入开发阶段...",
            height=100
        )

        if st.button("保存项目信息"):
            st.success("项目信息已保存！现在可以开始对话了。")
            st.rerun()
    else:
        st.success("项目信息已设置")
        st.write("**项目背景：**")
        st.write(st.session_state.project_context[:100] + "..." if len(
            st.session_state.project_context) > 100 else st.session_state.project_context)
        st.write("**当前状态：**")
        st.write(st.session_state.current_status[:80] + "..." if len(
            st.session_state.current_status) > 80 else st.session_state.current_status)

# 主聊天区域
# 检查项目信息是否已设置
if not st.session_state.project_context or not st.session_state.current_status:
    st.warning("👈 请在左侧边栏先填写项目信息，然后开始对话。")
    st.stop()

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
user_input = st.chat_input("请输入你的反思请求或遇到的问题...")

if user_input:
    # 添加用户消息到历史并显示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成智能体回复
    with st.chat_message("assistant"):
        with st.spinner("元认知教练正在思考..."):
            # 调用链生成回复
            response = chain.invoke({
                "project_context": st.session_state.project_context,
                "current_status": st.session_state.current_status,
                "user_input": user_input
            })
            st.markdown(response)

    # 添加助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": response})

# 添加一些使用建议
with st.expander("💡 使用建议"):
    st.markdown("""
    **你可以这样提问：**
    - "我不知道这个设计方案是否合适..."
    - "我们的项目遇到瓶颈了，不知道下一步该怎么走..."
    - "刚刚完成了项目的第一阶段，想做个总结反思..."
    - "团队成员对方案有分歧，不知道该怎么决策..."
    - "感觉学习收获不够明显，想梳理一下..."

    **元认知教练会：**
    1. 根据你的情境选择合适的反思模式
    2. 提出启发性的问题引导你思考
    3. 提供实用的反思工具和框架
    4. 鼓励你从经验中学习和成长
    """)