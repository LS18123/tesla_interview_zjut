import json
import os
import shutil
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class TeslaVectorStoreManager:
    def __init__(self):
        """
        初始化向量库管理器，配置路径、硬件加速及 Embedding 模型
        """
        # 1. 强制镜像源，确保在国内网络环境下稳定下载 BGE-M3 模型
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 2. 基础路径配置
        self.base_dir = r"D:\tesla_interview"
        # 核心修改：改为读取 PDR 架构生成的子块文件
        self.chunks_path = os.path.join(self.base_dir, "tesla_child_chunks.json")
        self.db_dir = os.path.join(self.base_dir, "tesla_db")

        # 3. 确定运行设备 (自动检测 CUDA)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" 正在初始化语义引擎...")
        print(f" 当前运行设备: {self.device.upper()}")

        if self.device == "cpu":
            print(" 提示: 未检测到显卡加速环境，将使用 CPU 运行。如果已有显卡，请检查 PyTorch 是否为 CUDA 版本。")

        # 4. 初始化精英级 Embedding 引擎 (BGE-M3)
        # BGE-M3 支持 8192 Token 长度，在处理财务分块时具有极高的语义密度
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}  # 开启归一化，提升余弦相似度计算精度
        )

    def _clean_old_db(self):
        """
        物理删除旧索引，防止由于父子架构变动（元数据结构变化）导致检索污染
        """
        if os.path.exists(self.db_dir):
            print(f"检测到旧版数据库，正在执行深度清理: {self.db_dir}")
            shutil.rmtree(self.db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

    def build_index(self):
        """
        构建 PDR 架构下的向量索引
        """
        if not os.path.exists(self.chunks_path):
            print(f"[ERROR] 找不到子块文件: {self.chunks_path}，请先运行新的 chunking.py")
            return

        # 执行清理操作
        self._clean_old_db()

        print("正在加载 PDR 语义子块数据...")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [c["page_content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        print(f"开始构建父子索引架构，目标容量: {len(texts)} 个子分块...")

        # 根据 GPU 显存建议 batch_size 设置为 1000
        batch_size = 1000 if self.device == "cuda" else 500
        vectorstore = None

        # 分批写入 Chroma，防止内存溢出和写入超时
        for i in tqdm(range(0, len(texts), batch_size), desc="向量化进度"):
            batch_texts = texts[i: i + batch_size]
            batch_metas = metadatas[i: i + batch_size]

            if vectorstore is None:
                # 首次创建并持久化
                vectorstore = Chroma.from_texts(
                    texts=batch_texts,
                    embedding=self.embeddings,
                    metadatas=batch_metas,
                    persist_directory=self.db_dir
                )
            else:
                # 增量添加
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metas
                )

        print(f"\n[SUCCESS] 向量库子块索引构建完成！")
        print(f"[PATH] 存储位置: {self.db_dir}")
        return vectorstore

    def verify_index(self, test_query="Automotive gross margin in 2023"):
        """
        验证性检索：重点检查 parent_id 是否成功存入元数据
        """
        print(f"\n--- 索引检索压力测试: [{test_query}] ---")

        # 加载持久化后的数据库
        vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

        # 模拟检索 Top 2 片段
        results = vectorstore.similarity_search_with_score(test_query, k=2)

        for i, (doc, score) in enumerate(results):
            # Chroma 的 score 为距离，越小表示越匹配
            print(f"【匹配项 {i + 1} | 距离评分: {score:.4f}】")
            m = doc.metadata
            print(f"坐标: {m.get('year')} {m.get('quarter')} | Parent_ID: {m.get('parent_id')}")
            print(f"分块预览: {doc.page_content[:100]}...")
            print("-" * 40)


if __name__ == "__main__":
    # 逻辑闭环：实例化、索引、验证
    manager = TeslaVectorStoreManager()
    manager.build_index()
    manager.verify_index()