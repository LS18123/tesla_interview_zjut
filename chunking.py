import json
import os
import copy
import hashlib
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TeslaPDRChunker:
    """
    特斯拉财报父子索引分块器 (Parent-Document Retrieval Chunker)
    """

    def __init__(self, child_size=400, child_overlap=50):
        # 基础路径配置
        self.base_dir = r"D:\tesla_interview"
        self.input_path = os.path.join(self.base_dir, "parsed_tesla_reports.json")

        # 输出两个文件：一个是给向量库的“子块”，一个是给回溯用的“父文档库”
        self.child_output_path = os.path.join(self.base_dir, "tesla_child_chunks.json")
        self.parent_output_path = os.path.join(self.base_dir, "tesla_parent_store.json")

        # 子块切分策略：较小的 size 能让向量特征更鲜明，减少语义漂移
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\nITEM", "\nItem", "\n\n", "\n|", "\n", " ", ""]
        )

    def _is_table_dense(self, text):
        """判断分块是否包含密集表格特征"""
        return text.count('|') > 5 or text.count('---') > 1

    def process(self):
        """
        执行父子索引层级的构建任务
        """
        if not os.path.exists(self.input_path):
            print(f"[ERROR] 找不到解析后的 JSON 文件，请确保已运行 parser.py")
            return

        print(" 正在构建父子索引层级架构...")
        with open(self.input_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        final_children = []
        parent_store = {}

        for page in tqdm(pages, desc="PDR Processing"):
            raw_text = page.get("text", "")
            meta = page.get("metadata", {})

            # 1. 生成父文档 ID (以每页财报为独立单位)
            # 格式示例: 2023_10-K_p45
            parent_id = f"{meta.get('source')}_p{meta.get('page')}"

            # 2. 存储父文档：保留完整的页面文本，解决“表格被腰斩”的根本问题
            parent_store[parent_id] = {
                "text": raw_text,
                "metadata": meta
            }

            # 3. 构造子块的面包屑 (Breadcrumbs)
            # 注入时空坐标，增强向量检索在特定年份/页码的敏感度
            breadcrumb = f"[{meta.get('year')} {meta.get('quarter')} | P{meta.get('page')}] "

            # 4. 清理并切分子块
            # 去掉 Parser 注入的装饰性 Context Header，只保留有效正文进行切分
            clean_content = raw_text.split("##############################")[-1].strip()
            if not clean_content:
                continue

            child_splits = self.child_splitter.split_text(clean_content)

            for i, split_text in enumerate(child_splits):
                # 将面包屑前缀直接压入子块文本，让 Embedding 学习到年份信息
                enriched_text = breadcrumb + split_text

                # 深度克隆元数据并绑定父节点 ID
                child_meta = copy.deepcopy(meta)
                child_meta["parent_id"] = parent_id  # 核心连接字段
                child_meta["chunk_id"] = f"{parent_id}_c{i}"
                child_meta["is_table"] = self._is_table_dense(split_text)

                final_children.append({
                    "page_content": enriched_text,
                    "metadata": child_meta
                })

        # 5. 持久化输出：向量库子块
        print(f"正在保存 {len(final_children)} 个子块到向量索引源...")
        with open(self.child_output_path, "w", encoding="utf-8") as f:
            json.dump(final_children, f, ensure_ascii=False, indent=2)

        # 6. 持久化输出：父文档回溯库
        print(f"正在构建父文档回溯字典...")
        with open(self.parent_output_path, "w", encoding="utf-8") as f:
            json.dump(parent_store, f, ensure_ascii=False, indent=2)

        print(f"\n[SUCCESS] 工业级 PDR 数据层构建完成！")
        print(f" 统计报告:")
        print(f" - 原始页面 (父文档): {len(parent_store)} 个")
        print(f" - 语义子块 (向量索引): {len(final_children)} 个")
        print(f" - 存储路径 (子块): {self.child_output_path}")
        print(f" - 存储路径 (父库): {self.parent_output_path}")


if __name__ == "__main__":
    chunker = TeslaPDRChunker()
    chunker.process()