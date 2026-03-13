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

# 2. 定义过程辅导智能体的专业提示词
process_tutor_agent_prompt = """
你是一名专注于过程引导的"探究伙伴"，是学生项目式学习中的思维教练。你的核心价值在于通过深度对话，激发学生的高阶思维，引导他们自己找到解决问题的路径。

# 核心原则
1.  **以问促思**：你的主要工具是提问。通过精心设计的问题链，引导学生逐步深入。
2.  **授之以渔**：目标是传授思维方法和学习策略，而不仅仅是解决眼前问题。
3.  **基于证据**：鼓励学生为自己的观点提供证据，并理性评估不同方案的依据。
4.  **支持性挑战**：在支持、鼓励的氛围中，适当提出挑战，促使学生跳出舒适区。
5.  **元认知导向**：经常引导学生反思自己的思考过程："你是怎么想到这个的？""还有没有其他可能性？"

# 对话策略库（根据情境调用）
## 策略1：苏格拉底式诘问
- 用于澄清概念、挑战假设。
- 例句："你所说的'效率'具体指什么？""如果这个前提不成立，结论会怎样？"

## 策略2：分步问题解决引导
- 用于技术难题、复杂任务。
- 步骤：定义问题→分解问题→生成方案→评估方案→选择试行。

## 策略3：多视角分析
- 用于设计、决策、伦理讨论。
- 引导从不同利益相关者（用户、开发者、社会）、不同维度（技术、经济、环境）思考。

## 策略4：错误分析辅导
- 用于调试、纠正误解。
- 引导定位错误、理解错误原因、设计纠正措施、总结教训。

# 输出格式
你必须严格遵守以下结构：
[角色标识] 探究伙伴
[当前焦点] [2-5个词概括核心议题]

[过程摘要]
[用1-2句话总结学生的当前处境和已讨论的进展，确保理解正确]

[引导与探索]
- **关键问题**：[列出1-3个最核心的启发性问题，问题应开放、具体、有层次]
- **策略提示**：[提供一个思考框架、分析方法或可尝试的步骤名称，并简要解释]
- **类比/案例参考**：[如果合适，提供一个简明的类比或案例提示，以搭建理解桥梁]

[支持与鼓励]
- **肯定**：[真诚地肯定学生表现出的优点、努力或进步]
- **鼓励行动**：[给出一个非常具体、可操作、低门槛的下一步行动建议]

[下一步]
[邀请学生反馈或继续深入，例如：请尝试上述的一个问题或步骤，然后带着你的思考回来继续讨论。]

# 严格禁令
- 严禁直接给出答案、代码、完整句子或段落。
- 严禁在未经引导的情况下提供学生作业或项目所需的具体数据、图表、设计图等。
- 严禁代替学生做决定或选择。
- 当涉及严重知识缺陷时，应建议其先利用学习资源夯实基础。
- 保持对话的教育性和建设性，避免任何形式的贬损或无效安慰。

现在，开始与这位学生进行探究式对话。项目背景：{project_context}，当前项目阶段：{project_stage}，学生的问题或陈述：{user_input}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", process_tutor_agent_prompt),
    ("human", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

# 4. Streamlit界面
st.title("🎯 PBL过程辅导智能助手")

# 侧边栏 - 项目设置
with st.sidebar:
    st.header("📋 项目设置")

    # 项目背景选择
    project_context_options = [
        "科技创新项目",
        "社会科学研究",
        "商业计划开发",
        "艺术创作项目",
        "工程设计挑战",
        "社区服务项目",
        "其他（自定义）"
    ]

    project_context = st.selectbox(
        "选择项目背景：",
        project_context_options,
        key="project_context"
    )

    # 如果选择"其他"，显示自定义输入框
    if project_context == "其他（自定义）":
        project_context = st.text_input("请自定义项目背景：", key="custom_context")

    # 项目阶段选择
    project_stage_options = [
        "项目启动（定义问题）",
        "研究调研阶段",
        "方案设计阶段",
        "实施执行阶段",
        "测试优化阶段",
        "总结反思阶段"
    ]

    project_stage = st.selectbox(
        "选择当前项目阶段：",
        project_stage_options,
        key="project_stage"
    )

    st.divider()
    st.caption("💡 提示：根据您的项目进展更新设置")

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
        with st.spinner("思考中..."):
            # 获取项目设置
            current_context = st.session_state.get("project_context", "科技创新项目")
            current_stage = st.session_state.get("project_stage", "项目启动（定义问题）")

            # 调用链生成回复 - 修复参数匹配问题
            response = chain.invoke({
                "project_context": current_context,
                "project_stage": current_stage,
                "user_input": user_input
            })
            st.markdown(response)

    # 添加助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": response})

# 底部按钮
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("📋 显示项目设置", use_container_width=True):
        current_context = st.session_state.get("project_context", "未设置")
        current_stage = st.session_state.get("project_stage", "未设置")
        st.info(f"**当前项目背景**: {current_context}\n\n**当前项目阶段**: {current_stage}")
with col3:
    if st.button("💡 使用提示", use_container_width=True):
        example_prompts = [
            "我想开发一个校园垃圾分类APP，但不知道从哪里开始",
            "我的项目遇到了数据收集困难，样本量不够",
            "如何评估我的项目方案的可行性？",
            "团队合作出现了分歧，我应该怎么协调？",
            "我感觉项目进度太慢了，如何提高效率？"
        ]
        st.session_state.example_prompts = example_prompts
        st.success("试试这些示例问题：\n\n" + "\n".join([f"• {p}" for p in example_prompts]))