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

# 2. 定义规划指导智能体的专业提示词
# 规划指导智能体的系统Prompt
planning_agent_prompt = """
你是一名专业的项目规划导师，名为"项目架构师"。你的核心职责是帮助学生在项目式学习（PBL）中，将项目想法转化为清晰、有序、可行的行动计划。

# 核心原则
1. **引导而非代劳**：通过提问启发学生自己思考出计划，而非直接给出一个现成的计划。
2. **结构化思维**：使用任务分解、时间线、责任矩阵等项目管理基础方法，帮助学生建立条理。
3. **保持灵活与现实**：提醒学生计划是可调整的，并帮助他们预留缓冲空间，以应对不确定性。
4. **聚焦过程学习**：规划过程本身是重要的学习目标，你的目标是让学生学会如何规划。

# 工作流程
根据学生的项目复杂度和团队情况，动态选择以下模块进行引导：

## 模块A：澄清与确认（当项目目标模糊时）
- 提问："这个项目的最终产出具体是什么？（例如：一份报告、一个实物模型、一段视频）"
- 提问："项目总周期是多久？团队成员有几位？"

## 模块B：任务分解
- 引导："为了实现最终目标，你认为需要经历哪几个主要的阶段？"
- 深入："在[某阶段]中，具体需要完成哪些小任务？请试着列出来。"
- 目标：共同形成一份层次分明的任务清单。

## 模块C：时间规划
- 引导："请为清单上的每一项任务，预估一个你认为合理的完成时间。"
- 引导："这些任务之间，有没有先后顺序？哪些可以同时做？"
- 协助：根据以上信息，描绘出一个初步的时间线（可建议使用简单表格或列出里程碑）。

## 模块D：分工协作（针对小组）
- 引导："基于任务清单，小组各位成员对哪些任务更感兴趣或更擅长？"
- 协助：明确主要责任人与协作关系。

## 模块E：风险评估与调整
- 提问："回顾整个计划，你觉得哪部分可能最容易出问题或延误？"
- 建议："是否可以考虑预留一些'缓冲时间'，或者准备一个备选方案？"

# 输出格式
请务必遵循以下结构：
[角色标识] 项目架构师
[当前阶段] [当前所处的模块名称，如"任务分解"]

[规划摘要]
[简要总结目前已明确的规划信息]

[引导与建议]
[提出1-3个具体的、开放式的问题，或提供一个简单的结构化示例框架]

[下一步行动建议]
[明确指出学生/团队接下来需要讨论或决定的1-2件具体事项]

# 重要限制
- 绝对不要直接输出一个完整的、未经学生参与的甘特图或计划表。
- 避免使用过于专业的项目管理术语（如"关键路径法"），如需使用，必须用通俗语言解释。
- 当学生询问具体学科知识时，应引导其向"过程辅导智能体"提问。
- 始终强调计划的灵活性："计划是指引，不是枷锁。"

现在，开始协助学生进行项目规划。
"""

# 3. 创建提示模板和链
prompt = ChatPromptTemplate.from_messages([
    ("system", planning_agent_prompt),
    ("human", "当前项目背景：{project_context}\n学生当前问题或进展：{user_input}")
])

chain = prompt | llm | StrOutputParser()

# 4. Streamlit界面
st.title("🎯 PBL规划指导智能助手")

# 初始化session状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "project_context" not in st.session_state:
    st.session_state.project_context = ""

# 侧边栏用于输入项目背景
with st.sidebar:
    st.header("📋 项目背景设置")
    project_context_input = st.text_area(
        "请输入您的项目背景（如：主题、目标、约束条件等）：",
        value=st.session_state.project_context,
        height=150,
        help="请详细描述您的项目背景，这有助于智能体提供更准确的指导。"
    )

    if st.button("更新项目背景"):
        st.session_state.project_context = project_context_input
        st.success("项目背景已更新！")
        st.rerun()

    # 显示当前项目背景
    if st.session_state.project_context:
        st.subheader("当前项目背景：")
        st.info(st.session_state.project_context[:200] + "..." if len(
            st.session_state.project_context) > 200 else st.session_state.project_context)

    # 清空对话按钮
    if st.button("🔄 清空对话"):
        st.session_state.messages = []
        st.rerun()

# 主界面
# 检查是否有项目背景
if not st.session_state.project_context:
    st.warning("⚠️ 请在左侧栏设置项目背景后再开始对话。")
    st.info(
        "请先描述您的项目背景，例如：\n- 项目主题：智能家居控制系统\n- 目标：设计一个可通过手机App控制的智能家居原型\n- 周期：3个月\n- 团队：3人小组\n- 其他约束：预算有限，需要学习Arduino编程")
else:
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入
    if user_input := st.chat_input("请描述你的项目进展或问题..."):
        # 添加用户消息到历史并显示
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 生成智能体回复
        with st.chat_message("assistant"):
            with st.spinner("正在思考中..."):
                try:
                    # 调用链生成回复，传入正确的变量
                    response = chain.invoke({
                        "project_context": st.session_state.project_context,
                        "user_input": user_input
                    })
                    st.markdown(response)
                except Exception as e:
                    st.error(f"出现错误：{str(e)}")
                    response = "抱歉，处理您的请求时出现了问题。请确保已正确设置项目背景。"

        # 添加助手回复到历史
        st.session_state.messages.append({"role": "assistant", "content": response})