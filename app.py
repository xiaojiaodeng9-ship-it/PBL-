"""
多智能体PBL支持系统完整实现
作者：[您的名字]
日期：2025年7月
描述：基于LangGraph实现多智能体协同的项目式学习支持系统
"""

import os
import json
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

# LangChain核心组件
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# Streamlit界面
import streamlit as st

# 设置页面
st.set_page_config(
    page_title="多智能体PBL支持系统",
    page_icon="🎯",
    layout="wide"
)


# ============================================
# 第一部分：配置管理和LLM初始化
# ============================================

class LLMConfig:
    """LLM配置管理类，优先从环境变量(.env文件)读取"""

    @staticmethod
    def load_env_config():
        """加载并显示环境变量配置"""
        # 环境变量优先级：1. .env文件 2. 系统环境变量
        config = {
            "api_key": os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("AI_API_BASE") or os.getenv("OPENAI_API_BASE"),
            "model": os.getenv("AI_MODEL") or "gpt-4o-mini",
            "source": "环境变量(.env文件)"
        }

        # 检查是否从.env文件加载
        if not config["api_key"]:
            config["source"] = "未找到配置"

        return config

    @staticmethod
    def get_llm():
        """获取配置好的LLM实例"""
        try:
            # 首先尝试从环境变量(.env文件)加载配置
            env_config = LLMConfig.load_env_config()

            api_key = env_config["api_key"]
            base_url = env_config["base_url"]
            model = env_config["model"]

            # 如果环境变量中没有，再尝试从Streamlit secrets获取
            if not api_key:
                try:
                    api_key = st.secrets.get("AI_API_KEY", "")
                    base_url = st.secrets.get("AI_API_BASE", "")
                    model = st.secrets.get("AI_MODEL", "gpt-4o-mini")
                    env_config["source"] = "Streamlit Secrets"
                except:
                    pass

            # 如果还没有，尝试从session_state获取（用户手动输入）
            if not api_key:
                api_key = st.session_state.get("ai_api_key", "")
                base_url = st.session_state.get("ai_api_base", "")
                model = st.session_state.get("ai_model", "gpt-4o-mini")
                if api_key:
                    env_config["source"] = "手动输入"

            if not api_key:
                st.warning("⚠️ 请设置AI API密钥。您可以在侧边栏输入，或者在.env文件中配置。")
                return None

            # 配置LLM参数
            llm_kwargs = {
                "model": model,
                "temperature": 0.3,
                "api_key": api_key,
            }

            # 如果有自定义base_url，则添加
            if base_url:
                llm_kwargs["base_url"] = base_url

            # 创建LLM实例
            llm = ChatOpenAI(**llm_kwargs)

            # 测试连接（可选，但推荐）
            try:
                # 简单的测试调用，但只做一次，避免频繁调用
                if not st.session_state.get("llm_test_passed", False):
                    test_response = llm.invoke("Hello")
                    if hasattr(test_response, 'content'):
                        st.session_state.llm_test_passed = True
                        # 显示配置来源
                        st.sidebar.success(f"✅ 使用{env_config['source']}配置")
                        if base_url:
                            st.sidebar.info(f"🌐 API地址: {base_url}")
                        st.sidebar.info(f"🤖 模型: {model}")
            except Exception as e:
                st.error(f"❌ LLM连接测试失败: {str(e)}")
                # 不立即返回None，可能只是测试问题，实际可用
                # 但标记测试失败
                st.session_state.llm_test_passed = False

            return llm

        except Exception as e:
            st.error(f"LLM初始化错误: {str(e)}")
            return None

    @staticmethod
    def get_collaboration_llm():
        """获取协作整合用的LLM实例（单独配置）"""
        try:
            # 使用与主LLM相同的配置
            return LLMConfig.get_llm()
        except:
            # 如果失败，尝试创建默认实例
            try:
                env_config = LLMConfig.load_env_config()
                return ChatOpenAI(
                    model=env_config["model"],
                    temperature=0.3,
                    api_key=env_config["api_key"],
                    base_url=env_config["base_url"]
                )
            except:
                return None


# ============================================
# 第二部分：五个智能体的核心提示词与链定义
# ============================================

class PBLAgents:
    """五个PBL智能体的实现"""

    def __init__(self, llm):
        self.llm = llm
        self._initialize_agents()

    def _initialize_agents(self):
        """初始化五个智能体链"""

        # 1. 立项智能体
        self.project_initiation_prompt = """
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

        现在，开始帮助这位学生吧！
        当前对话历史：{chat_history}
        学生当前问题：{user_input}
        """

        # 2. 规划智能体
        self.planning_agent_prompt = """
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
        当前项目背景：{project_context}
        学生当前问题或进展：{user_input}
        """

        # 3. 资源智能体
        self.resource_agent_prompt = """
        你是"资源导航员"，是项目式学习中的脚手架与工具箱管理者。你的使命是赋能学生，让他们能自主获取所需的知识与工具。

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

        现在，开始为这位学生提供支持。
        项目背景：{project_context}
        当前项目阶段：{project_stage}
        学生具体请求：{user_input}
        """

        # 4. 辅导智能体（简化版）
        self.coaching_agent_prompt = """
        你是一名专业的PBL过程辅导导师，名为"苏格拉底引导者"。你的核心职责是通过启发式提问，引导学生自主探索、解决项目执行中遇到的具体问题，而不是直接给出答案。

        # 核心原则
        1. **引导而非告知**：使用苏格拉底式提问法，帮助学生自己发现答案。
        2. **聚焦思维过程**：关注"如何思考"而不仅仅是"正确答案"。
        3. **提供适度支架**：在学生完全卡住时，提供思考框架或简化问题。
        4. **连接知识与实践**：帮助学生将理论知识应用于项目实际问题。

        # 工作方式
        根据问题类型采用不同策略：

        ## 策略A：概念理解问题
        - 提问："你能用自己的话解释一下这个概念吗？"
        - 追问："这个概念和你已经知道的哪些知识有联系？"
        - 举例："你能举一个现实生活中的例子来说明吗？"

        ## 策略B：问题解决方法
        - 引导："我们先明确一下，这个问题的核心是什么？"
        - 分解："能不能把这个大问题拆解成几个小问题？"
        - 追溯："你之前尝试过哪些方法？结果如何？"

        ## 策略C：设计决策问题
        - 提问："这个设计方案试图解决什么具体问题？"
        - 权衡："不同的设计方案各有什么优缺点？"
        - 验证："你如何验证这个设计的有效性？"

        ## 策略D：技术实现问题
        - 引导："你理解的实现步骤是什么？"
        - 诊断："你遇到的具体错误信息或现象是什么？"
        - 资源："你需要的是具体的代码示例，还是理解背后的原理？"

        # 输出格式
        [角色标识] 苏格拉底引导者
        [问题类型] [概念理解/问题解决/设计决策/技术实现]

        [引导性问题]
        1. [第一个开放性问题]
        2. [第二个深入性问题]
        3. [第三个应用性问题]

        [思考框架]
        - [提供一个简化的思考路径或检查清单]

        [下一步探索]
        [建议学生接下来可以尝试的具体行动或思考方向]

        # 重要限制
        - 绝不直接给出完整答案或完整代码。
        - 一次提问不超过3个问题，避免认知过载。
        - 当涉及专业知识深度不足时，建议学生查阅"资源智能体"推荐的学习资料。
        - 保持耐心和支持性语气。

        现在，开始引导这位学生。
        当前项目：{project_context}
        学生问题：{user_input}
        """

        # 5. 评价智能体
        self.evaluation_agent_prompt = """
        你是"元认知教练"，一位精通反思性实践与形成性评估的导师。你的存在是为了让学习变得可见，让成长有迹可循。

        # 核心信念
        1. 最有价值的评估来自学习者自身。你是提问者，而非裁判。
        2. 反思是连接经验与学习的桥梁。你帮助搭建这座桥。
        3. 错误和困境是深度学习的最佳素材。你引导将其转化为资源。

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

        现在，开始你的教练对话。
        项目背景：{project_context}
        当前阶段/成果：{current_status}
        学生发起的反思请求或触发点：{user_input}
        """

        # 创建各个智能体的链
        self._create_agent_chains()

    def _create_agent_chains(self):
        """为每个智能体创建LangChain链"""

        # 立项智能体链
        self.initiation_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", self.project_initiation_prompt),
                    ("human", "{user_input}")
                ])
                | self.llm
                | StrOutputParser()
        )

        # 规划智能体链
        self.planning_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", self.planning_agent_prompt),
                    ("human", "{user_input}")
                ])
                | self.llm
                | StrOutputParser()
        )

        # 资源智能体链
        self.resource_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", self.resource_agent_prompt),
                    ("human", "{user_input}")
                ])
                | self.llm
                | StrOutputParser()
        )

        # 辅导智能体链
        self.coaching_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", self.coaching_agent_prompt),
                    ("human", "{user_input}")
                ])
                | self.llm
                | StrOutputParser()
        )

        # 评价智能体链
        self.evaluation_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", self.evaluation_agent_prompt),
                    ("human", "{user_input}")
                ])
                | self.llm
                | StrOutputParser()
        )


# ============================================
# 第三部分：LangGraph状态与路由定义
# ============================================

class AgentState(TypedDict):
    """定义多智能体系统的状态"""
    user_input: str
    project_context: Optional[Dict[str, Any]]
    current_stage: Optional[str]
    agent_outputs: Dict[str, str]
    active_agents: List[str]
    final_response: Optional[str]
    messages: Annotated[List[Any], add_messages]
    routing_decision: Optional[Dict]
    need_collaboration: bool
    chat_history: str


class TaskRouter:
    """智能路由决策器"""

    def __init__(self, llm):
        self.llm = llm
        self._create_routing_chain()

    def _create_routing_chain(self):
        """创建路由决策链"""
        routing_prompt = """
        你是一个智能路由决策器，负责分析学生问题并分配到最合适的PBL智能体。

        可用的智能体：
        1. initiation_agent (立项智能体): 负责项目选题、驱动性问题设计
        2. planning_agent (规划智能体): 负责项目规划、时间管理、任务分解
        3. resource_agent (资源智能体): 负责推荐学习资源、工具、资料
        4. coaching_agent (辅导智能体): 负责过程指导、答疑解惑
        5. evaluation_agent (评价智能体): 负责评价反馈、反思总结

        请分析学生的问题，决定调用哪个智能体最合适。

        学生问题: {user_input}
        当前项目阶段: {current_stage}
        项目背景: {project_context}

        返回JSON格式：
        {{
            "primary_agent": "agent_name",
            "secondary_agents": ["agent1", "agent2"],
            "reasoning": "简要说明为什么选择这个智能体",
            "need_collaboration": true/false
        }}
        """

        self.routing_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", routing_prompt),
                    ("human", "{user_input}")
                ])
                | self.llm
                | StrOutputParser()
        )

    def decide_routing(self, state: AgentState) -> Dict:
        """做出路由决策"""
        try:
            # 构建上下文
            context = {
                "user_input": state["user_input"],
                "current_stage": state.get("current_stage", "立项"),
                "project_context": str(state.get("project_context", {}))
            }

            # 调用路由链
            response = self.routing_chain.invoke(context)

            # 尝试解析JSON
            try:
                decision = json.loads(response)
            except:
                # 如果解析失败，使用默认决策
                decision = self._rule_based_routing(state)

            return decision

        except Exception as e:
            print(f"路由决策错误: {e}")
            return self._rule_based_routing(state)

    def _rule_based_routing(self, state: AgentState) -> Dict:
        """基于规则的路由决策（备用）"""
        user_input = state["user_input"].lower()

        # 关键词映射
        initiation_keywords = ["想法", "主题", "选题", "做什么", "兴趣", "点子"]
        planning_keywords = ["计划", "时间", "分工", "步骤", "安排", "怎么开始"]
        resource_keywords = ["资源", "资料", "教程", "工具", "参考", "找"]
        coaching_keywords = ["怎么做", "为什么", "帮助", "指导", "问题", "困难", "不会"]
        evaluation_keywords = ["评价", "反思", "总结", "反馈", "怎么样", "改进"]

        # 检查关键词
        for keyword in initiation_keywords:
            if keyword in user_input:
                return {
                    "primary_agent": "initiation_agent",
                    "secondary_agents": [],
                    "reasoning": f"检测到关键词'{keyword}'，需要立项支持",
                    "need_collaboration": False
                }

        for keyword in planning_keywords:
            if keyword in user_input:
                return {
                    "primary_agent": "planning_agent",
                    "secondary_agents": [],
                    "reasoning": f"检测到关键词'{keyword}'，需要规划支持",
                    "need_collaboration": False
                }

        for keyword in resource_keywords:
            if keyword in user_input:
                return {
                    "primary_agent": "resource_agent",
                    "secondary_agents": [],
                    "reasoning": f"检测到关键词'{keyword}'，需要资源支持",
                    "need_collaboration": False
                }

        for keyword in evaluation_keywords:
            if keyword in user_input:
                return {
                    "primary_agent": "evaluation_agent",
                    "secondary_agents": [],
                    "reasoning": f"检测到关键词'{keyword}'，需要评价支持",
                    "need_collaboration": False
                }

        # 默认使用辅导智能体
        return {
            "primary_agent": "coaching_agent",
            "secondary_agents": [],
            "reasoning": "未识别到特定关键词，使用默认辅导智能体",
            "need_collaboration": False
        }


# ============================================
# 第四部分：智能体节点函数
# ============================================

def create_agent_nodes(pbl_agents, task_router):
    """创建五个智能体的节点函数"""

    def task_router_node(state: AgentState) -> AgentState:
        """任务路由节点"""
        print(f"路由节点：分析用户输入: {state['user_input'][:50]}...")

        # 做出路由决策
        decision = task_router.decide_routing(state)
        state["routing_decision"] = decision
        state["need_collaboration"] = decision.get("need_collaboration", False)

        # 设置激活的智能体
        primary = decision["primary_agent"]
        secondary = decision.get("secondary_agents", [])
        state["active_agents"] = [primary] + secondary

        print(f"路由决策：主要智能体={primary}, 协作={state['need_collaboration']}")

        return state

    def project_initiation_node(state: AgentState) -> AgentState:
        """立项智能体节点"""
        print("调用立项智能体...")

        # 构建上下文
        context = {
            "chat_history": state.get("chat_history", ""),
            "user_input": state["user_input"]
        }

        # 调用智能体链
        try:
            response = pbl_agents.initiation_chain.invoke(context)
            state["agent_outputs"]["initiation"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))

            # 尝试提取项目信息
            if "项目名称" in response or "主题" in response:
                # 这里可以添加项目信息提取逻辑
                pass

        except Exception as e:
            error_msg = f"立项智能体错误: {str(e)}"
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def planning_node(state: AgentState) -> AgentState:
        """规划智能体节点"""
        print("调用规划智能体...")

        # 构建上下文
        context = {
            "project_context": str(state.get("project_context", {})),
            "user_input": state["user_input"]
        }

        try:
            response = pbl_agents.planning_chain.invoke(context)
            state["agent_outputs"]["planning"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))
        except Exception as e:
            error_msg = f"规划智能体错误: {str(e)}"
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def resource_node(state: AgentState) -> AgentState:
        """资源智能体节点"""
        print("调用资源智能体...")

        # 构建上下文
        context = {
            "project_context": str(state.get("project_context", {})),
            "project_stage": state.get("current_stage", "执行"),
            "user_input": state["user_input"]
        }

        try:
            response = pbl_agents.resource_chain.invoke(context)
            state["agent_outputs"]["resource"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))
        except Exception as e:
            error_msg = f"资源智能体错误: {str(e)}"
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def coaching_node(state: AgentState) -> AgentState:
        """辅导智能体节点"""
        print("调用辅导智能体...")

        # 构建上下文
        context = {
            "project_context": str(state.get("project_context", {})),
            "user_input": state["user_input"]
        }

        try:
            response = pbl_agents.coaching_chain.invoke(context)
            state["agent_outputs"]["coaching"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))
        except Exception as e:
            error_msg = f"辅导智能体错误: {str(e)}"
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def evaluation_node(state: AgentState) -> AgentState:
        """评价智能体节点"""
        print("调用评价智能体...")

        # 构建上下文
        context = {
            "project_context": str(state.get("project_context", {})),
            "current_status": state.get("current_stage", "评价"),
            "user_input": state["user_input"]
        }

        try:
            response = pbl_agents.evaluation_chain.invoke(context)
            state["agent_outputs"]["evaluation"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))
        except Exception as e:
            error_msg = f"评价智能体错误: {str(e)}"
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def collaboration_integrator_node(state: AgentState) -> AgentState:
        """协作整合节点（多个智能体输出整合）"""
        print("调用协作整合器...")

        # 收集所有智能体输出
        outputs = state["agent_outputs"]

        if not outputs:
            state["final_response"] = "暂时没有智能体输出需要整合。"
            return state

        # 如果有多个输出，进行整合
        if len(outputs) > 1:
            # 构建整合提示
            integration_prompt = f"""
            请将以下多个智能体的建议整合成一个连贯、全面的回答：

            {chr(10).join([f'【{agent}】: {output}' for agent, output in outputs.items()])}

            学生的问题: {state['user_input']}

            请整合以上建议，要求：
            1. 去除重复信息
            2. 保持逻辑连贯
            3. 突出关键建议
            4. 语言自然流畅

            整合后的回答：
            """

            # 使用LLM进行整合
            try:
                llm = LLMConfig.get_collaboration_llm()
                if llm:
                    response = llm.invoke(integration_prompt).content
                    state["final_response"] = response
                else:
                    # 如果LLM获取失败，直接拼接
                    combined = "\n\n".join([f"【{agent}】\n{output}" for agent, output in outputs.items()])
                    state["final_response"] = f"综合多个智能体的建议：\n\n{combined}"
            except Exception as e:
                # 如果整合失败，直接拼接
                combined = "\n\n".join([f"【{agent}】\n{output}" for agent, output in outputs.items()])
                state["final_response"] = f"综合多个智能体的建议：\n\n{combined}"
        else:
            # 只有一个输出，直接使用
            state["final_response"] = list(outputs.values())[0]

        state["messages"].append(AIMessage(content=state["final_response"]))
        return state

    # 返回节点函数字典
    return {
        "task_router": task_router_node,
        "initiation": project_initiation_node,
        "planning": planning_node,
        "resource": resource_node,
        "coaching": coaching_node,
        "evaluation": evaluation_node,
        "collaboration_integrator": collaboration_integrator_node
    }


# ============================================
# 第五部分：构建LangGraph工作流
# ============================================

def build_multi_agent_workflow(llm):
    """构建多智能体协同工作流"""

    # 初始化智能体和路由器
    pbl_agents = PBLAgents(llm)
    task_router = TaskRouter(llm)

    # 创建节点函数
    nodes = create_agent_nodes(pbl_agents, task_router)

    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加所有节点
    for node_name, node_func in nodes.items():
        workflow.add_node(node_name, node_func)

    # 设置入口点
    workflow.set_entry_point("task_router")

    # 定义路由逻辑（从路由节点到具体智能体）
    def route_after_router(state: AgentState) -> str:
        """根据路由决策选择下一个节点"""
        routing = state.get("routing_decision", {})

        if not routing:
            return "coaching"  # 默认

        # 检查是否需要协作
        if routing.get("need_collaboration", False):
            return "collaboration_integrator"
        else:
            # 映射智能体名称到节点名称
            agent_map = {
                "initiation_agent": "initiation",
                "planning_agent": "planning",
                "resource_agent": "resource",
                "coaching_agent": "coaching",
                "evaluation_agent": "evaluation"
            }

            primary = routing.get("primary_agent", "coaching_agent")
            return agent_map.get(primary, "coaching")

    # 添加条件边
    workflow.add_conditional_edges(
        "task_router",
        route_after_router,
        {
            "initiation": "initiation",
            "planning": "planning",
            "resource": "resource",
            "coaching": "coaching",
            "evaluation": "evaluation",
            "collaboration_integrator": "collaboration_integrator"
        }
    )

    # 所有智能体节点都指向END
    for node_name in ["initiation", "planning", "resource", "coaching", "evaluation", "collaboration_integrator"]:
        workflow.add_edge(node_name, END)

    # 编译工作流
    graph = workflow.compile()

    return graph, pbl_agents


# ============================================
# 第六部分：Streamlit界面
# ============================================

def initialize_session_state():
    """初始化Streamlit会话状态"""
    # 确保 project_context 存在
    if "project_context" not in st.session_state:
        st.session_state.project_context = {
            "project_name": "未命名项目",
            "stage": "立项",
            "description": "",
            "start_date": datetime.now().strftime("%Y-%m-%d")
        }

    # 确保其他必要的 session_state 变量存在
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    # 初始化工作流（如果还没有）
    if "workflow" not in st.session_state or "agents" not in st.session_state:
        try:
            # 获取 LLM 实例
            llm = LLMConfig.get_llm()

            if llm is None:
                # 显示环境变量信息
                env_config = LLMConfig.load_env_config()
                if env_config["api_key"]:
                    st.warning(f"⚠️ LLM初始化失败，但检测到.env文件配置。请检查API密钥和地址是否正确。")
                    st.info(f"当前配置来源: {env_config['source']}")
                    if env_config['base_url']:
                        st.info(f"API地址: {env_config['base_url']}")
                else:
                    st.warning("⚠️ 未检测到API配置。请在.env文件中配置或侧边栏输入。")
                return

            # 构建工作流
            workflow, agents = build_multi_agent_workflow(llm)

            st.session_state.workflow = workflow
            st.session_state.agents = agents
            st.success("✅ 系统初始化成功！")

        except Exception as e:
            st.error(f"初始化智能体系统时出错: {e}")


def main():
    """主程序入口"""
    # 初始化会话状态（必须在任何访问 session_state 之前调用）
    initialize_session_state()

    # 设置标题和介绍
    st.title("🎯 多智能体PBL支持系统")
    st.markdown("""
    本系统整合了五个专业智能体，为您的项目式学习提供全方位支持：
    1. **项目催化剂** - 帮助立项选题
    2. **项目架构师** - 协助规划安排
    3. **资源导航员** - 推荐学习资源
    4. **苏格拉底引导者** - 提供过程辅导
    5. **元认知教练** - 引导评价反思
    """)

    # 显示环境变量状态
    env_config = LLMConfig.load_env_config()
    if env_config["api_key"]:
        st.sidebar.success(f"✅ 已从{env_config['source']}加载配置")
    else:
        st.sidebar.warning("⚠️ 未找到.env配置")

    # 侧边栏 - 项目设置和API配置
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # API配置部分（当环境变量不存在时显示）
        if not env_config["api_key"]:
            with st.expander("🔑 AI API 配置", expanded=True):
                st.info("支持OpenAI、Close AI等兼容API")

                # 从session_state获取当前值
                current_key = st.session_state.get("ai_api_key", "")
                current_base = st.session_state.get("ai_api_base", "")
                current_model = st.session_state.get("ai_model", "gpt-4o-mini")

                # API密钥输入
                api_key = st.text_input(
                    "API密钥",
                    value=current_key,
                    type="password",
                    help="输入您的AI API密钥"
                )

                # API基础地址（可选）
                base_url = st.text_input(
                    "API基础地址（可选）",
                    value=current_base,
                    help="例如：https://api.closeai.com/v1 （留空使用默认地址）"
                )

                # 模型选择
                model = st.selectbox(
                    "模型",
                    ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
                    index=0,
                    help="选择要使用的模型"
                )

                # 保存配置按钮
                if st.button("保存API配置", type="primary"):
                    if api_key:
                        st.session_state.ai_api_key = api_key
                        st.session_state.ai_api_base = base_url if base_url else ""
                        st.session_state.ai_model = model

                        # 更新环境变量（当前会话）
                        os.environ["AI_API_KEY"] = api_key
                        if base_url:
                            os.environ["AI_API_BASE"] = base_url
                        os.environ["AI_MODEL"] = model

                        st.success("✅ API配置已保存！")
                        st.info("请刷新页面以重新初始化系统。")
                        st.rerun()
                    else:
                        st.warning("请输入API密钥")
        else:
            # 显示当前配置信息
            with st.expander("📋 当前配置信息", expanded=True):
                key_display = env_config["api_key"][:8] + "..." + env_config["api_key"][-4:] if len(
                    env_config["api_key"]) > 12 else "***"
                st.success(f"✅ 已从{env_config['source']}加载配置")
                st.text(f"🔑 API密钥: {key_display}")
                if env_config["base_url"]:
                    st.text(f"🌐 API地址: {env_config['base_url']}")
                st.text(f"🤖 模型: {env_config['model']}")

                # 重新加载配置按钮
                if st.button("🔄 重新加载.env配置"):
                    # 重新加载环境变量
                    load_dotenv(override=True)
                    st.success("✅ 已重新加载.env文件")
                    st.rerun()

        st.divider()

        # 项目设置
        st.header("📋 项目设置")

        # 安全地访问 project_context
        project_context = st.session_state.get("project_context", {})

        project_name = st.text_input("项目名称",
                                     value=project_context.get("project_name", "未命名项目"))

        # 安全地获取当前阶段
        current_stage = project_context.get("stage", "立项")
        stage_options = ["立项", "规划", "执行", "评价"]
        try:
            stage_index = stage_options.index(current_stage)
        except ValueError:
            stage_index = 0

        project_stage = st.selectbox(
            "当前阶段",
            stage_options,
            index=stage_index
        )

        project_desc = st.text_area("项目描述",
                                    value=project_context.get("description", ""))

        if st.button("更新项目信息"):
            st.session_state.project_context.update({
                "project_name": project_name,
                "stage": project_stage,
                "description": project_desc
            })
            st.success("项目信息已更新！")

        st.divider()

        # 系统状态
        st.subheader("🔍 系统状态")

        # 显示工作流状态
        if "workflow" in st.session_state:
            st.success("✅ 工作流已初始化")
        else:
            st.warning("❌ 工作流未初始化")

        # 显示消息数量
        st.text(f"💬 消息数量: {len(st.session_state.get('messages', []))}")

        # 测试连接按钮
        if st.button("🔗 测试API连接"):
            llm = LLMConfig.get_llm()
            if llm:
                try:
                    with st.spinner("测试连接中..."):
                        test_response = llm.invoke("Hello, are you working?")
                    st.success("✅ API连接正常！")
                except Exception as e:
                    st.error(f"❌ 连接失败: {str(e)}")
            else:
                st.warning("请先配置API密钥")

        st.divider()

        # 显示调试信息（可选）
        with st.expander("🐛 调试信息"):
            st.write("项目上下文:", st.session_state.project_context)
            st.write("环境变量配置:", env_config)

        # 清空聊天历史按钮
        if st.button("🔄 开始新的对话"):
            st.session_state.messages = []
            st.session_state.chat_history = ""
            st.rerun()

    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        messages = st.session_state.get("messages", [])
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
            else:
                # 如果是AIMessage或HumanMessage对象
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                content = message.content if hasattr(message, 'content') else str(message)

            with st.chat_message(role):
                st.markdown(content)

    # 聊天输入
    if prompt := st.chat_input("请输入您关于PBL项目的问题..."):
        # 检查系统是否初始化
        if "workflow" not in st.session_state:
            st.error("系统未初始化，请先配置API密钥。")
            return

        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 准备状态
        initial_state = AgentState(
            user_input=prompt,
            project_context=st.session_state.project_context,
            current_stage=st.session_state.project_context.get("stage", "立项"),
            agent_outputs={},
            active_agents=[],
            final_response=None,
            messages=[],
            routing_decision=None,
            need_collaboration=False,
            chat_history=st.session_state.get("chat_history", "")
        )

        # 显示思考状态
        with st.chat_message("assistant"):
            with st.spinner("🤔 多智能体正在协同思考中..."):
                try:
                    # 运行工作流
                    result = st.session_state.workflow.invoke(initial_state)

                    # 获取响应
                    response = result.get("final_response", "抱歉，暂时无法回答您的问题。")

                    # 显示响应
                    st.markdown(response)

                    # 添加到消息历史
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # 更新聊天历史（简化版本）
                    chat_entry = f"用户: {prompt}\n助手: {response}\n\n"
                    st.session_state.chat_history += chat_entry

                    # 显示使用的智能体
                    active_agents = result.get("active_agents", [])
                    if active_agents:
                        agent_names = {
                            "initiation_agent": "项目催化剂",
                            "planning_agent": "项目架构师",
                            "resource_agent": "资源导航员",
                            "coaching_agent": "苏格拉底引导者",
                            "evaluation_agent": "元认知教练"
                        }

                        used_agents = [agent_names.get(agent, agent) for agent in active_agents]
                        st.caption(f"🔧 本次使用了: {', '.join(used_agents)}")

                except Exception as e:
                    error_msg = f"系统处理时出现错误: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # 底部信息
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("💡 **使用提示**\n\n尝试询问：\n• '我想做环保项目，有什么建议？'\n• '如何规划项目时间？'\n• '推荐一些学习资源'")

    with col2:
        st.success("🔄 **工作流程**\n\n1. 输入问题\n2. 智能路由分析\n3. 调用合适智能体\n4. 返回专业建议")

    with col3:
        st.warning("⚠️ **注意事项**\n\n• 支持.env文件配置API\n• 支持OpenAI/Close AI等API\n• 建议结合教师指导")


# ============================================
# 第七部分：运行程序
# ============================================

if __name__ == "__main__":
    # 运行主程序
    main()