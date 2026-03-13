"""
多智能体PBL支持系统完整实现
作者：邓小娇
日期：2025年12月
描述：基于LangGraph实现多智能体协同的项目式学习支持系统
（增强上下文记忆，适度提供示例，支持用户账号登录）
"""

import os
import json
import time
import re
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

# 密码加密
from werkzeug.security import generate_password_hash, check_password_hash

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
# 用户认证与数据存储配置
# ============================================
USERS_FILE = "users.json"          # 存储用户名和密码哈希
USER_DATA_DIR = "user_data"        # 每个用户的数据文件存放目录

# 确保用户数据目录存在
os.makedirs(USER_DATA_DIR, exist_ok=True)

def load_users():
    """加载用户数据库"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """保存用户数据库"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def get_user_data_file(username):
    """获取指定用户的数据文件路径"""
    return os.path.join(USER_DATA_DIR, f"{username}.json")

def load_chat_data(username):
    """加载指定用户的聊天数据"""
    file_path = get_user_data_file(username)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载用户数据失败: {e}")
    # 默认数据
    return {
        "project_context": {
            "project_name": "未命名项目",
            "stage": "立项",
            "description": "",
            "start_date": datetime.now().strftime("%Y-%m-%d")
        },
        "messages": [],
        "chat_history": ""
    }

def save_chat_data(username, project_context, messages, chat_history):
    """保存指定用户的聊天数据"""
    file_path = get_user_data_file(username)
    try:
        data = {
            "project_context": project_context,
            "messages": messages,
            "chat_history": chat_history
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存用户数据失败: {e}")

# ============================================
# 登录/注册界面
# ============================================
def login_page():
    """显示登录页面"""
    st.title("🔐 用户登录")

    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        col1, col2 = st.columns(2)
        with col1:
            login_submitted = st.form_submit_button("登录")
        with col2:
            go_to_register = st.form_submit_button("注册新账号")

    if login_submitted:
        users = load_users()
        if username in users and check_password_hash(users[username], password):
            st.session_state.authenticated = True
            st.session_state.username = username
            # 加载该用户的数据
            data = load_chat_data(username)
            st.session_state.project_context = data["project_context"]
            st.session_state.messages = data["messages"]
            st.session_state.chat_history = data["chat_history"]
            st.rerun()
        else:
            st.error("用户名或密码错误")

    if go_to_register:
        st.session_state.show_register = True
        st.rerun()

def register_page():
    """显示注册页面"""
    st.title("📝 用户注册")

    with st.form("register_form"):
        new_username = st.text_input("用户名")
        new_password = st.text_input("密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        col1, col2 = st.columns(2)
        with col1:
            register_submitted = st.form_submit_button("注册")
        with col2:
            back_to_login = st.form_submit_button("返回登录")

    if register_submitted:
        if not new_username or not new_password:
            st.error("用户名和密码不能为空")
        elif new_password != confirm_password:
            st.error("两次输入的密码不一致")
        else:
            users = load_users()
            if new_username in users:
                st.error("用户名已存在")
            else:
                # 存储新用户
                users[new_username] = generate_password_hash(new_password)
                save_users(users)
                # 自动登录
                st.session_state.authenticated = True
                st.session_state.username = new_username
                # 初始化该用户的数据（默认）
                st.session_state.project_context = {
                    "project_name": "未命名项目",
                    "stage": "立项",
                    "description": "",
                    "start_date": datetime.now().strftime("%Y-%m-%d")
                }
                st.session_state.messages = []
                st.session_state.chat_history = ""
                # 保存初始数据
                save_chat_data(new_username,
                              st.session_state.project_context,
                              st.session_state.messages,
                              st.session_state.chat_history)
                st.success("注册成功！")
                st.rerun()

    if back_to_login:
        st.session_state.show_register = False
        st.rerun()

# ============================================
# 第一部分：配置管理和LLM初始化
# ============================================

class LLMConfig:
    """LLM配置管理类，优先从环境变量(.env文件)读取"""

    @staticmethod
    def load_env_config():
        """加载环境变量配置"""
        # 环境变量优先级：1. .env文件 2. 系统环境变量
        config = {
            "api_key": os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("AI_API_BASE") or os.getenv("OPENAI_API_BASE"),
            "model": os.getenv("AI_MODEL") or "gpt-4o-mini",
            "source": "环境变量(.env文件)"
        }

        return config

    @staticmethod
    def get_llm():
        """获取配置好的LLM实例"""
        try:
            # 从环境变量(.env文件)加载配置
            env_config = LLMConfig.load_env_config()

            api_key = env_config["api_key"]
            base_url = env_config["base_url"]
            model = env_config["model"]

            if not api_key:
                print("⚠️ 未检测到API配置，请检查.env文件")
                return None

            # 配置LLM参数
            llm_kwargs = {
                "model": model,
                "temperature": 0.3,
                "api_key": api_key,
                "max_retries": 2,
                "timeout": 30,
            }

            # 如果有自定义base_url，则添加
            if base_url:
                llm_kwargs["base_url"] = base_url
                # 移除末尾的斜杠（如果有）
                if base_url.endswith('/'):
                    llm_kwargs["base_url"] = base_url.rstrip('/')

            # 创建LLM实例
            llm = ChatOpenAI(**llm_kwargs)

            print(f"✅ LLM实例创建成功，模型: {model}")
            if base_url:
                print(f"🌐 API地址: {base_url}")

            return llm

        except Exception as e:
            print(f"❌ LLM配置错误: {str(e)[:100]}")
            return None

    @staticmethod
    def get_collaboration_llm():
        """获取协作整合用的LLM实例（单独配置）"""
        try:
            # 使用与主LLM相同的配置
            return LLMConfig.get_llm()
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
        - **如果需要，可以提供一个相关的驱动性问题示例，如：'例如，如果你对环保感兴趣，可以问“如何减少校园塑料垃圾？”'**
        - 目标：形成清晰的、可探究的项目问题

        ## 阶段4：可行性评估（当问题明确后）
        - 评估项目规模、资源需求、时间框架
        - 提供简化或扩展的建议
        - **如果学生感到无从下手，可以提供一个简化的项目规模参考示例，如：'一个适合一周内完成的小项目可以聚焦在调查校园内某类垃圾的数量并提出改进建议。'**
        - 目标：确保项目在约束条件下可行

        # 输出格式
        请严格按照以下格式组织回复：
        [角色标识] 项目催化剂
        [当前阶段] [阶段名称]

        [主要内容]
        [你的引导性问题或建议，必要时包含示例]

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
        - **如果学生难以开始，可以提供一个简化的任务分解示例，例如：'比如一个关于“校园垃圾分类”的项目，可以分为调研、分析、设计解决方案、制作展示物等阶段。' 并强调“这只是一个参考，你需要根据自己的项目调整。”**
        - 目标：共同形成一份层次分明的任务清单。

        ## 模块C：时间规划
        - 引导："请为清单上的每一项任务，预估一个你认为合理的完成时间。"
        - 引导："这些任务之间，有没有先后顺序？哪些可以同时做？"
        - **如果学生需要，可以提供一个简单的时间表示例，如：'可以用表格形式列出任务、负责人、开始日期、结束日期。例如：| 任务 | 负责人 | 开始 | 结束 |'** 
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
        [提出1-3个具体的、开放式的问题，或提供一个简单的结构化示例框架，必要时包含示例]

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
        当前对话历史：{chat_history}
        """

        # 3. 资源智能体（已修改为优先推荐国内开源资源，并确保资源可用性）
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

        # 新增要求：优先推荐国内开源资源，且确保资源当前可访问
        在满足需求的前提下，请优先推荐**国内的开源/免费资源**，并确保推荐的网站或资源**目前可正常访问**。以下是一些经过验证、当前可用的国内资源平台示例（可根据实际情况灵活选择，不限于此列表）：

        ## 学习类平台（免费/开源）
        - **B站 (Bilibili)**：www.bilibili.com - 被称为"中国人的在线大学"，有大量高校公开课、技术教程、科普视频，免费且互动性强 
        - **中国大学MOOC**：www.icourse163.org - 汇聚国内顶尖高校的优质课程，覆盖全学科，所有课程均可免费观看学习 
        - **学堂在线**：www.xuetangx.com - 源自清华大学的慕课平台，尤其以理工科和计算机科学课程见长，与国际edX平台合作 
        - **网易公开课**：open.163.com - 资源库庞大，包含国内外名校公开课、TED演讲、可汗学院课程等 

        ## 代码托管与技术社区
        - **Gitee（码云）**：gitee.com - 国内知名的代码托管平台，沉淀了大量适合本土开发者的开源项目和中文技术文档 
        - **GitHub**：github.com - 全球最大的开源托管平台，虽然不是国内平台，但作为开源资源的核心库，仍可推荐 
        - **CSDN**：www.csdn.net - 全球知名中文IT技术交流平台，包含原创博客、问答、下载等丰富资源 
        - **掘金**：juejin.cn - 面向全球中文开发者的技术内容分享与交流平台，社区活跃 
        - **博客园**：www.cnblogs.com - 面向开发者的纯净技术交流社区，知识分享氛围好 
        - **51CTO博客**：blog.51cto.com - 专业程序员、运维/网络工程师的IT创作平台 
        - **SegmentFault（思否）**：segmentfault.com - 中国专业的开发者技术社区，以技术问答、技术博客为核心 
        - **知乎**：www.zhihu.com - 中文问答社区，科技、编程等领域有大量高质量问答 

        ## 开发者社区与工具
        - **阿里云开发者社区**：developer.aliyun.com - 覆盖云计算、大数据、人工智能等技术领域，提供分享、学习、工具资源 
        - **腾讯云开发者社区**：cloud.tencent.com/developer - 腾讯云官方技术分享社区，汇聚云计算使用和开发经验 
        - **独立开发很酷 (SoloDev)**：www.solodev.cool - 专注于服务独立开发者的综合性社区平台，提供技术交流、产品展示、资源对接 

        ## AI开发与模型资源
        - **AtomGit**：atomgit.com - 国内AI开源社区，提供模型在线体验、开源项目托管，如GLM-5、Qwen3.5等模型可在平台体验 
        - **和鲸社区**：www.heywhale.com - 数据科学和AI开发者社区，提供免费数据集、Notebook环境、项目实践
        - **百度AI Studio**：aistudio.baidu.com - 百度提供的AI学习与实训平台，免费提供GPU算力、数据集和课程

        ## 设计/创意资源（免费）
        - **Pixso**：pixso.cn - 国产在线设计平台，提供免费组件库、图标包和UI模板 
        - **即时设计**：js.design - 国产在线UI设计工具，社区有大量免费资源可复用 

        # 响应策略（根据诊断结果选择）
        ## 策略A：针对知识概念不清
        - **资源类型**：科普动画、图文解析、互动模拟、百科词条。
        - **脚手架**："先看视频建立直观印象，再精读文字厘清定义。"

        ## 策略B：针对技能/方法不会
        - **资源类型**：分步教程、代码范例、模板下载、工具官方文档。
        - **脚手架**："参照这个模板的结构，填入你自己的内容。"或"先完成教程里的第一个小练习。"
        - **如果需要，可以提供一个简化的思考框架示例，如：'比如你想学习如何做用户调研，可以按照“明确目的-设计问卷-收集数据-分析结果”的步骤进行。'**

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
        - **思考/行动框架**：[提供一个简化的步骤、问题列表或分析角度，必要时包含示例]
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
        当前对话历史：{chat_history}
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
        - **如果学生仍然困惑，可以提供一个简化的思考框架示例，如：'比如理解“光合作用”，可以想象植物利用阳光、水和二氧化碳制造食物。'**

        ## 策略B：问题解决方法
        - 引导："我们先明确一下，这个问题的核心是什么？"
        - 分解："能不能把这个大问题拆解成几个小问题？"
        - 追溯："你之前尝试过哪些方法？结果如何？"
        - **如果学生卡住，可以提供一个简化的思考框架示例，如：'比如你想提高社区垃圾分类参与率，可以先调查原因，再设计宣传方案，最后测试效果。'**

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
        - [提供一个简化的思考路径或检查清单，必要时包含示例]

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
        当前对话历史：{chat_history}
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
          - **如果学生不清楚如何评估，可以提供一个简化的评估标准示例，如：'例如，调研阶段可以看：是否收集了至少10份问卷？是否采访了2位相关人士？'**

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
        - 如需，提供一个极其简洁的**反思模板**或**评估清单**，必要时包含示例。

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
        当前对话历史：{chat_history}
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
        当前对话历史: {chat_history}

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
                "project_context": str(state.get("project_context", {})),
                "chat_history": state.get("chat_history", "")
            }

            # 调用路由链
            response = self.routing_chain.invoke(context)

            # 尝试解析JSON
            try:
                # 清理响应，确保它是有效的JSON
                response_clean = response.strip()
                # 如果响应以```json开头和结尾，移除这些标记
                if response_clean.startswith("```json"):
                    response_clean = response_clean[7:]
                if response_clean.endswith("```"):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()

                decision = json.loads(response_clean)

                # 验证决策格式
                required_keys = ["primary_agent", "secondary_agents", "reasoning"]
                if not all(key in decision for key in required_keys):
                    raise ValueError("路由决策缺少必要字段")

                # 确保secondary_agents是列表
                if not isinstance(decision.get("secondary_agents", []), list):
                    decision["secondary_agents"] = []

                # 确保need_collaboration是布尔值
                if "need_collaboration" not in decision:
                    decision["need_collaboration"] = False

                return decision

            except json.JSONDecodeError as e:
                # 尝试提取JSON部分
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        decision = json.loads(json_match.group())
                        return decision
                    except:
                        pass
                return self._rule_based_routing(state)
            except Exception as e:
                return self._rule_based_routing(state)

        except Exception as e:
            return self._rule_based_routing(state)

    def _rule_based_routing(self, state: AgentState) -> Dict:
        """基于规则的路由决策（备用）"""
        user_input = state["user_input"].lower()

        # 扩展关键词映射
        initiation_keywords = ["想法", "主题", "选题", "做什么", "兴趣", "点子", "项目", "做什么项目", "创意"]
        planning_keywords = ["计划", "规划", "时间", "分工", "步骤", "安排", "怎么开始", "制定", "详细计划", "项目计划", "时间表", "任务"]
        resource_keywords = ["资源", "资料", "教程", "工具", "参考", "找", "学习", "教材", "书籍", "网站", "软件"]
        coaching_keywords = ["怎么做", "为什么", "帮助", "指导", "问题", "困难", "不会", "请教", "如何解决", "疑问"]
        evaluation_keywords = ["评价", "反思", "总结", "反馈", "怎么样", "改进", "评估", "检查", "回顾", "分析"]

        # 检查关键词 - 规划相关
        for keyword in planning_keywords:
            if keyword in user_input:
                return {
                    "primary_agent": "planning_agent",
                    "secondary_agents": [],
                    "reasoning": f"检测到关键词'{keyword}'，需要规划支持",
                    "need_collaboration": False
                }

        # 检查关键词 - 资源相关
        for keyword in resource_keywords:
            if keyword in user_input:
                return {
                    "primary_agent": "resource_agent",
                    "secondary_agents": [],
                    "reasoning": f"检测到关键词'{keyword}'，需要资源支持",
                    "need_collaboration": False
                }

        # 检查关键词 - 立项相关
        for keyword in initiation_keywords:
            if keyword in user_input:
                return {
                    "primary_agent": "initiation_agent",
                    "secondary_agents": [],
                    "reasoning": f"检测到关键词'{keyword}'，需要立项支持",
                    "need_collaboration": False
                }

        # 检查关键词 - 评价相关
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
        print(f"路由节点：分析用户输入: {state['user_input'][:100]}...")

        # 做出路由决策
        decision = task_router.decide_routing(state)
        state["routing_decision"] = decision
        state["need_collaboration"] = decision.get("need_collaboration", False)

        # 设置激活的智能体
        primary = decision["primary_agent"]
        secondary = decision.get("secondary_agents", [])
        state["active_agents"] = [primary] + secondary

        print(f"路由决策：主要智能体={primary}, 次要智能体={secondary}, 协作={state['need_collaboration']}")

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

            print(f"立项智能体响应长度: {len(response)} 字符")

        except Exception as e:
            error_msg = f"立项智能体错误: {str(e)}"
            state["agent_outputs"]["initiation"] = error_msg
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def planning_node(state: AgentState) -> AgentState:
        """规划智能体节点"""
        print("调用规划智能体...")

        # 构建上下文
        project_context = state.get("project_context", {})
        if isinstance(project_context, dict):
            project_info = f"""
项目名称: {project_context.get('project_name', '未命名项目')}
项目阶段: {project_context.get('stage', '规划')}
项目描述: {project_context.get('description', '无')}
用户问题: {state['user_input']}
            """
        else:
            project_info = str(project_context)

        context = {
            "project_context": project_info,
            "user_input": state["user_input"],
            "chat_history": state.get("chat_history", "")
        }

        try:
            response = pbl_agents.planning_chain.invoke(context)
            state["agent_outputs"]["planning"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))

            print(f"规划智能体响应长度: {len(response)} 字符")

        except Exception as e:
            error_msg = f"规划智能体错误: {str(e)}"
            state["agent_outputs"]["planning"] = error_msg
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
            "user_input": state["user_input"],
            "chat_history": state.get("chat_history", "")
        }

        try:
            response = pbl_agents.resource_chain.invoke(context)
            state["agent_outputs"]["resource"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))

            print(f"资源智能体响应长度: {len(response)} 字符")

        except Exception as e:
            error_msg = f"资源智能体错误: {str(e)}"
            state["agent_outputs"]["resource"] = error_msg
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def coaching_node(state: AgentState) -> AgentState:
        """辅导智能体节点"""
        print("调用辅导智能体...")

        # 构建上下文
        context = {
            "project_context": str(state.get("project_context", {})),
            "user_input": state["user_input"],
            "chat_history": state.get("chat_history", "")
        }

        try:
            response = pbl_agents.coaching_chain.invoke(context)
            state["agent_outputs"]["coaching"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))

            print(f"辅导智能体响应长度: {len(response)} 字符")

        except Exception as e:
            error_msg = f"辅导智能体错误: {str(e)}"
            state["agent_outputs"]["coaching"] = error_msg
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
            "user_input": state["user_input"],
            "chat_history": state.get("chat_history", "")
        }

        try:
            response = pbl_agents.evaluation_chain.invoke(context)
            state["agent_outputs"]["evaluation"] = response
            state["final_response"] = response
            state["messages"].append(AIMessage(content=response))

            print(f"评价智能体响应长度: {len(response)} 字符")

        except Exception as e:
            error_msg = f"评价智能体错误: {str(e)}"
            state["agent_outputs"]["evaluation"] = error_msg
            state["final_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))

        return state

    def collaboration_integrator_node(state: AgentState) -> AgentState:
        """协作整合节点（多个智能体输出整合）"""
        print("调用协作整合器...")

        # 收集所有智能体输出
        outputs = state["agent_outputs"]

        if not outputs or len(outputs) == 0:
            print("警告：没有智能体输出需要整合")
            # 如果没有输出，尝试使用规则路由选择主智能体
            routing = state.get("routing_decision", {})
            primary = routing.get("primary_agent", "coaching_agent")

            # 根据主智能体直接调用对应节点
            if primary == "initiation_agent":
                return project_initiation_node(state)
            elif primary == "planning_agent":
                return planning_node(state)
            elif primary == "resource_agent":
                return resource_node(state)
            elif primary == "evaluation_agent":
                return evaluation_node(state)
            else:
                return coaching_node(state)

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
# 第六部分：主程序（含登录验证）
# ============================================

def initialize_session_state():
    """初始化Streamlit会话状态（不含数据加载，数据在登录后加载）"""
    # 仅初始化工作流相关
    if "workflow" not in st.session_state or "agents" not in st.session_state:
        try:
            # 获取 LLM 实例
            llm = LLMConfig.get_llm()

            if llm is None:
                st.warning("⚠️ 系统初始化失败，请检查.env文件中的API配置")
                return

            # 构建工作流
            workflow, agents = build_multi_agent_workflow(llm)

            st.session_state.workflow = workflow
            st.session_state.agents = agents
            st.session_state.llm = llm

            print("✅ 系统初始化成功！")

        except Exception as e:
            print(f"初始化智能体系统时出错: {e}")

def logout():
    """登出，清除会话状态"""
    for key in ['authenticated', 'username', 'project_context', 'messages', 'chat_history']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.show_register = False
    st.rerun()

def main():
    """主程序入口"""
    # 初始化会话状态（工作流等）
    initialize_session_state()

    # 处理登录状态
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.show_register = False

    # 如果未认证，显示登录/注册页面
    if not st.session_state.authenticated:
        if st.session_state.get('show_register', False):
            register_page()
        else:
            login_page()
        return

    # 已登录，显示主界面
    st.title("🎯 多智能体PBL支持系统")
    st.markdown("""
    本系统整合了五个专业智能体，为您的项目式学习提供全方位支持：
    1. **项目催化剂** - 帮助立项选题（提供示例参考）
    2. **项目架构师** - 协助规划安排（提供示例参考）
    3. **资源导航员** - 推荐学习资源（优先国内开源资源）
    4. **苏格拉底引导者** - 提供过程辅导（提供示例参考）
    5. **元认知教练** - 引导评价反思（提供示例参考）
    """)

    # 侧边栏 - 项目设置 + 登出
    with st.sidebar:
        st.header(f"👋 欢迎，{st.session_state.username}")
        if st.button("🚪 登出"):
            logout()

        st.divider()
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

        # 当项目描述更新时，自动更新项目上下文
        if project_desc and project_desc != project_context.get("description", ""):
            st.session_state.project_context["description"] = project_desc

        if st.button("更新项目信息"):
            st.session_state.project_context.update({
                "project_name": project_name,
                "stage": project_stage,
                "description": project_desc
            })
            # 保存到文件
            save_chat_data(st.session_state.username,
                          st.session_state.project_context,
                          st.session_state.messages,
                          st.session_state.chat_history)
            st.success("✅ 项目信息已更新！")
            st.rerun()

        st.divider()

        # 清空聊天历史按钮
        if st.button("🗑️ 清空聊天历史"):
            st.session_state.messages = []
            st.session_state.chat_history = ""
            # 清空文件
            save_chat_data(st.session_state.username,
                          st.session_state.project_context,
                          [],
                          "")
            st.success("✅ 聊天历史已清空！")
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
            st.error("❌ 系统未初始化，请检查.env文件中的API配置")
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
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 多智能体正在协同思考中...")

            try:
                # 运行工作流
                result = st.session_state.workflow.invoke(initial_state)

                # 获取响应
                response = result.get("final_response", "抱歉，暂时无法回答您的问题。")

                # 显示响应
                message_placeholder.markdown(response)

                # 添加到消息历史
                st.session_state.messages.append({"role": "assistant", "content": response})

                # 更新聊天历史
                chat_entry = f"用户: {prompt}\n助手: {response}\n\n"
                st.session_state.chat_history += chat_entry

                # 保存数据到文件
                save_chat_data(st.session_state.username,
                              st.session_state.project_context,
                              st.session_state.messages,
                              st.session_state.chat_history)

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
                error_msg = str(e)
                if "Connection error" in error_msg or "connect" in error_msg.lower():
                    error_msg = "网络连接失败，请检查API配置和网络连接。"
                elif "Invalid API key" in error_msg:
                    error_msg = "API密钥无效，请检查.env文件配置。"
                elif "rate limit" in error_msg.lower():
                    error_msg = "API调用频率限制，请稍后重试。"
                elif "timeout" in error_msg.lower():
                    error_msg = "请求超时，请检查网络连接或重试。"

                full_error = f"系统处理时出现错误: {error_msg}"
                message_placeholder.error(full_error)
                st.session_state.messages.append({"role": "assistant", "content": f"抱歉，处理时出现错误: {error_msg}"})
                # 发生错误时也保存（但可能不完整）
                save_chat_data(st.session_state.username,
                              st.session_state.project_context,
                              st.session_state.messages,
                              st.session_state.chat_history)

    # 底部信息 - 只保留使用提示
    st.divider()
    st.info("💡 **使用提示**：尝试询问：'我想做环保项目，有什么建议？'、'如何规划项目时间？'、'推荐一些学习资源'等")


# ============================================
# 第七部分：运行程序
# ============================================

if __name__ == "__main__":
    # 运行主程序
    main()