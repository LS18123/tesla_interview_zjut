import os
import json
import torch
import gradio as gr
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. 核心环境配置 (与 PDR 数据层严格对齐)
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DEEPSEEK_API_KEY = "" # 记得在此处填入您的真实 Key
BASE_URL = "https://api.deepseek.com"


class TeslaPDRQAStore:
    """
    基于父子索引 (PDR) 架构的特斯拉财报审计大脑
    """

    def __init__(self):
        self.base_dir = r"D:\tesla_interview"
        self.db_dir = os.path.join(self.base_dir, "tesla_db")
        # 核心回溯库路径
        self.parent_store_path = os.path.join(self.base_dir, "tesla_parent_store.json")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 初始化 BGE-M3 语义引擎 (必须与索引构建时严格一致)
        print(f" 正在同步 PDR 语义引擎 (Device: {self.device})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 加载向量库 (子块检索层)
        self.vector_db = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

        # 3. 加载父文档回溯字典 (PDR 的心脏)
        print(" 正在加载父文档上下文映射表...")
        if not os.path.exists(self.parent_store_path):
            raise FileNotFoundError(f"找不到父库文件: {self.parent_store_path}，请先运行 chunking.py")

        with open(self.parent_store_path, "r", encoding="utf-8") as f:
            self.parent_store = json.load(f)

        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

    # ==========================================
    # 2. 意图拆解：审计任务规划
    # ==========================================
    def _analyze_intent(self, question):
        planner_prompt = f"""
        你是一个特斯拉财务审计专家。请将用户的问题拆解为具体的检索任务 JSON。

        用户问题：{question}

        输出格式要求：
        {{
            "target_years": ["2021", "2023"],
            "finance_metrics": ["Gross Margin", "Automotive Revenue"],
            "audit_focus": "Item 8"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "你只输出 JSON。"},
                          {"role": "user", "content": planner_prompt}],
                response_format={'type': 'json_object'}
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            # 容错处理
            return {"target_years": [], "finance_metrics": [question]}

    # ==========================================
    # 3. PDR 执行层：精准命中 -> 完整回溯
    # ==========================================
    def _pdr_retrieve(self, audit_plan):
        """
        核心重构：利用 Parent_ID 召回整页内容
        """
        years = audit_plan.get("target_years", [])
        keywords = " ".join(audit_plan.get("finance_metrics", []))

        # 1. 检索 Top 子块 (Child Chunks)
        # 子块更短，语义匹配度更高
        child_docs = []
        if years:
            for year in years:
                # 强制元数据过滤，锁定年份范围
                docs = self.vector_db.similarity_search(
                    query=keywords,
                    k=3,
                    filter={"year": str(year)}
                )
                child_docs.extend(docs)
        else:
            child_docs = self.vector_db.similarity_search(keywords, k=8)

        # 2. 回溯父文档全文，并执行物理去重 (防止多子块命中同一页导致上下文重复)
        retrieved_p_ids = set()
        full_context_blocks = []

        for doc in child_docs:
            p_id = doc.metadata.get("parent_id")
            if p_id and p_id not in retrieved_p_ids:
                retrieved_p_ids.add(p_id)
                parent_data = self.parent_store.get(p_id)
                if parent_data:
                    # 组合成带来源标识的完整页上下文
                    block = (
                        f"### [审计证据页: {p_id}] ###\n"
                        f"{parent_data['text']}\n"
                        f"-----------------------------------"
                    )
                    full_context_blocks.append(block)

        return "\n\n".join(full_context_blocks)

    # ==========================================
    # 4. 深度审计推理层
    # ==========================================
    def answer(self, question):
        # 1. 生成规划
        plan = self._analyze_intent(question)
        print(f" 审计规划: {plan}")

        # 2. 执行 PDR 检索
        context = self._pdr_retrieve(plan)

        # 3. 最终生成分析报告
        system_prompt = """你现在是特斯拉特聘的高级财务审计分析师。
        你的任务是基于提供的【完整审计证据页】进行多年度对比分析。

        【准则】：
        1. 视野全局化：表格可能跨多行，请务必结合“Total Automotive Revenues”等汇总项进行计算。
        2. 来源精确化：每个数据点必须标注具体页码（如根据 2021 Q3, P35）。
        3. 处理口径：如果 2022 年报中对比了 2021 年的旧数据，请优先使用 2021 年原始季报中的披露数据。
        4. 诚实原则：若资料中确实没有提及特定指标，请诚实说明。
        """

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"审计任务：{question}\n\n审计资料库：\n{context}"}
            ],
            temperature=0.1  # 财务分析需要高一致性
        )
        return response.choices[0].message.content


# ==========================================
# 5. UI 交互界面 (Gradio)
# ==========================================
def launch():
    agent = TeslaPDRQAStore()

    with gr.Blocks(theme=gr.themes.Ocean(), title="Tesla PDR Auditor") as demo:
        gr.Markdown("#  特斯拉深度财报审计系统 (PDR 完全体)")
        gr.Markdown("当前模式：**父子索引映射 (Parent-Document Retrieval)** | 检索模型：BGE-M3")

        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="输入审计指令",
                    lines=5,
                    placeholder="例如：对比2021-2023年毛利率最高季度及当时的背景..."
                )
                btn = gr.Button("开始深度审计", variant="primary")
            with gr.Column(scale=2):
                output_md = gr.Markdown(label="审计报告分析")

        gr.Examples(
            examples=[
                ["对比2021-2023年，特斯拉在哪个季度的汽车毛利率最高？"],
                ["汇总2022年四个季度的研发费用，并对比2023年全年的数值。"],
                ["特斯拉在2023年财报中是如何描述供应链风险及产能瓶颈的？"]
            ],
            inputs=question_input
        )

        btn.click(fn=agent.answer, inputs=question_input, outputs=output_md)

    demo.launch(server_port=7860, share=False)


if __name__ == "__main__":
    launch()