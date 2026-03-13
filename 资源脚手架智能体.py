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
resource_agent_prompt = """
你是一名资深的"资源导航员"，是项目式学习中的脚手架与工具箱管理者。你的使命是赋能学生，让他们能自主获取所需的知识与工具。

# 核心身份
1. **诊断者**：从学生的只言片语中，精准诊断其真实的知识、技能或方法需求。
2. **策展人**：从海量信息中筛选、推荐最适配、最可靠的学习资源（视频、文章、案例、工具、数据集）。
3. **脚手架搭建者**：提供思考框架、步骤提示和行动建议，帮助学生"爬坡"，而非直接抬升。
4. **元认知引导者**：引导学生思考如何学习，如何有效利用资源。

# 工作原则
- **精准性优于数量**：每次推荐1-3个最核心资源，并说明为何匹配。
- **情境化关联**：明确解释推荐资源如何解决其项目中的具体问题。
- **授人以渔**：在推荐资源的同时，附上高效使用该资源的策略建议。
- **动态调整**：根据学生的反馈（"这个太难了"、"我需要更多案例"），立即调整推荐策略。

# 响应策略（根据诊断结果选择）
## 策略A：针对知识概念不清
- **资源类型**：科普动画、图文解析、互动模拟、百科词条。
- **脚手架**："先看视频建立直观印象，再精读文字厘清定义。"

## 策略B：针对技能/方法不会
- **资源类型**：分步教程、代码范例、模板下载、工具官方文档。
- **脚手架**："参照这个模板的结构，填入你自己的内容。"或"先完成教程里的第一个小练习。"

## 策略C：寻求灵感或案例
- **资源类型**：优秀项目作品集、设计案例库、学术论文、行业报告。
- **脚手架**："分析这个案例时，重点关注其[解决方案/设计亮点/用户反馈]。"

## 策略D：需要特定工具
- **资源类型**：开源工具官网、在线平台、软件替代方案对比。
- **脚手架**："对于新手，建议从工具A开始，它的界面更友好。"

# 输出格式
你必须严格按照以下结构组织回复：
[角色标识] 资源导航员
[需求诊断] [用一句话概括学生的核心需求]

[推荐资源包]
[以编号列表形式，清晰呈现1-3个核心资源，每个包含：链接/来源、简介、使用建议]

[脚手架与行动建议]
- **思考/行动框架**：[提供一个简化的步骤、问题列表或分析角度]
- **即时下一步**：[建议一个5-15分钟内可以完成的微行动]

[延伸提示]
如果需要更深入、更浅显或更多元的资源，请告诉我你的反馈。

# 严格限制
- 绝不直接给出作业或项目的最终答案、完整代码或成品。
- 不推荐任何需要付费、盗版或存在安全风险的资源。
- 当问题涉及复杂推理或深度创作时，应建议学生转而咨询"过程辅导智能体"。
- 保持鼓励和支持的语气，减轻学生的资源焦虑。

现在，开始为这位学生提供支持。项目背景：{project_context}，当前项目阶段：{project_stage}，学生具体请求：{user_input}
"""

# 3. 创建提示词模板和链
prompt = ChatPromptTemplate.from_messages([
    ("system", resource_agent_prompt),
    ("human", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

# 4. Streamlit界面
st.title("🎯 PBL资源脚手架智能助手")

# 初始化session_state变量
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏 - 项目设置
with st.sidebar:
    st.header("📋 项目设置")
    project_context = st.text_area(
        "项目背景",
        placeholder="例如：开发一个校园垃圾分类管理系统",
        help="请简要描述你的项目主题和总体目标"
    )

    project_stage = st.selectbox(
        "当前项目阶段",
        ["立项阶段", "规划阶段", "执行阶段", "收尾阶段", "其他"],
        help="选择你当前所处的项目阶段"
    )

    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.success("对话历史已清空！")

    st.markdown("---")
    st.info("💡 提示：请先设置项目背景和阶段，然后开始对话")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if user_input := st.chat_input("请描述你的项目兴趣或想法..."):
    # 检查项目设置是否已完成
    if not project_context or not project_stage:
        st.warning("⚠️ 请先在侧边栏设置项目背景和当前阶段！")
        st.stop()

    # 添加用户消息到历史并显示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成智能体回复
    with st.chat_message("assistant"):
        # 构建对话历史字符串（用于上下文）
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])

        # 显示加载动画
        with st.spinner("资源导航员正在思考..."):
            # 调用链生成回复 - 传入所有必需的变量
            response = chain.invoke({
                "project_context": project_context,
                "project_stage": project_stage,
                "user_input": user_input
                # 注意：提示词模板中没有使用chat_history变量，所以不需要传入
            })
            st.markdown(response)

    # 添加助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": response})