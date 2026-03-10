import pdfplumber
import pandas as pd
import re
import os
import json
import copy
from tqdm import tqdm


class TeslaReportParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        # 初始化基础元数据，确保键名与检查脚本严格一致
        self.metadata = {
            "year": "Unknown",
            "quarter": "Unknown",  # 统一使用 quarter
            "doc_type": "Unknown",
            "is_amendment": False,
            "source": self.filename,
            "page": 0,  # 统一使用 page
            "section": "Front Matter"
        }
        self.current_item = "Front Matter"

    def _extract_doc_info(self, pdf):
        """
        核心修复：通过截止日期强制锁定季度和文档类型，防止 10-K/Q 混淆
        """
        header_text = ""
        # 扫描前5页，足以覆盖所有封面信息
        for i in range(min(5, len(pdf.pages))):
            header_text += pdf.pages[i].extract_text() or ""

        upper_text = header_text.upper()

        # 1. 精准识别会计周期 (例如: September 30, 2023)
        date_pattern = r"(?:ENDED|ENDING)\s+([A-Z][a-z]+\s+\d{1,2},\s+(20\d{2}))"
        date_match = re.search(date_pattern, header_text, re.IGNORECASE)

        if date_match:
            full_date = date_match.group(1).upper()
            self.metadata["year"] = str(date_match.group(2))

            # --- 业务逻辑硬性绑定 ---
            if "DECEMBER 31" in full_date:
                self.metadata["quarter"] = "FY"
                self.metadata["doc_type"] = "10-K"
            elif "SEPTEMBER 30" in full_date:
                self.metadata["quarter"] = "Q3"
                self.metadata["doc_type"] = "10-Q"
            elif "JUNE 30" in full_date:
                self.metadata["quarter"] = "Q2"
                self.metadata["doc_type"] = "10-Q"
            elif "MARCH 31" in full_date:
                self.metadata["quarter"] = "Q1"
                self.metadata["doc_type"] = "10-Q"

        # 2. 识别修订版 (优先级最高，覆盖之前的 doc_type)
        if "FORM 10-K/A" in upper_text or "AMENDMENT" in upper_text:
            self.metadata["doc_type"] = "10-K/A"
            self.metadata["is_amendment"] = True
        elif "FORM 10-Q/A" in upper_text:
            self.metadata["doc_type"] = "10-Q/A"
            self.metadata["is_amendment"] = True

        # 3. 兜底逻辑：如果正文没匹配到年份，从文件名提取
        if self.metadata["year"] == "Unknown":
            fn_match = re.search(r"20(\d{2})", self.filename)
            if fn_match:
                self.metadata["year"] = "20" + fn_match.group(1)

    def _table_to_markdown(self, table):
        """
        将列表转换为标准的 Markdown 表格
        """
        if not table or len(table) < 1: return ""

        df = pd.DataFrame(table)
        df = df.dropna(how='all').fillna('')

        # 清洗单元格：去除空格、替换内部换行符
        try:
            # 兼容 Pandas 2.1+
            df = df.map(lambda x: str(x).replace('\n', ' ').strip())
        except AttributeError:
            df = df.applymap(lambda x: str(x).replace('\n', ' ').strip())

        # 过滤掉全空行
        df = df[~(df == '').all(axis=1)]
        if df.empty: return ""

        try:
            return df.to_markdown(index=False)
        except Exception:
            # 备用方案
            cols = df.columns.tolist()
            header = "| " + " | ".join([str(c) for c in cols]) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = ["| " + " | ".join([str(val) for val in row]) + " |" for _, row in df.iterrows()]
            return "\n".join([header, sep] + rows)

    def _detect_item_header(self, text):
        """
        识别 SEC 章节，增加长度限制防止抓取正文
        """
        # 正则：匹配 ITEM + 数字 + 标题文字 (限制在 100 字符内防止过度抓取)
        pattern = r"ITEM\s+([1-9][0-9]*[A-Z]?)\.\s+([A-Z\s,]{5,100})(?=\n|\.\s|[a-z])"
        match = re.search(pattern, text)
        if match:
            item_num = match.group(1)
            item_name = match.group(2).strip().split('\n')[0]
            return f"Item {item_num} ({item_name})"
        return None

    def parse(self):
        parsed_data = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                # 预提取全局属性
                self._extract_doc_info(pdf)

                for page_num, page in enumerate(pdf.pages):
                    # 安全裁剪：防止 Bounding Box 溢出
                    try:
                        tables = page.find_tables()
                        valid_bboxes = []
                        for t in tables:
                            b = t.bbox
                            safe_b = (max(0, b[0]), max(0, b[1]), min(page.width, b[2]), min(page.height, b[3]))
                            if safe_b[2] > safe_b[0] and safe_b[3] > safe_b[1]:
                                valid_bboxes.append(safe_b)

                        # 文本/表格分离
                        non_table_text_obj = page
                        for bbox in valid_bboxes:
                            non_table_text_obj = non_table_text_obj.outside_bbox(bbox)

                        clean_text = non_table_text_obj.extract_text() or ""
                    except Exception:
                        clean_text = page.extract_text() or ""
                        tables = []

                    # 实时同步元数据字典
                    new_item = self._detect_item_header(clean_text)
                    if new_item:
                        self.current_item = new_item

                    self.metadata["page"] = page_num + 1
                    self.metadata["section"] = self.current_item

                    # 提取表格
                    md_tables = []
                    for table_obj in tables:
                        raw_table = table_obj.extract()
                        md_str = self._table_to_markdown(raw_table)
                        if md_str: md_tables.append(md_str)

                    # 构造 Context Header
                    context_header = (
                        f"### [SEC_REPORT_CONTEXT] ###\n"
                        f"Source: {self.metadata['source']}\n"
                        f"Document: {self.metadata['year']} {self.metadata['doc_type']} ({self.metadata['quarter']})\n"
                        f"Current Section: {self.metadata['section']}\n"
                        f"Page Number: {self.metadata['page']}\n"
                        f"Amendment: {'Yes' if self.metadata['is_amendment'] else 'No'}\n"
                        f"##############################\n\n"
                    )

                    combined_content = clean_text + "\n\n" + "\n\n".join(md_tables)

                    if combined_content.strip():
                        # 深拷贝，确保每一页的元数据独立
                        parsed_data.append({
                            "text": context_header + combined_content,
                            "metadata": copy.deepcopy(self.metadata)
                        })

        except Exception as e:
            print(f"\n[ERROR] Failed to parse {self.filename}: {str(e)}")

        return parsed_data


def batch_process(input_dir, output_file):
    if not os.path.exists(input_dir):
        print(f"Error: Path {input_dir} not found.")
        return

    all_results = []
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF files. Deep parsing started...")

    for f in tqdm(pdf_files, desc="Parsing"):
        file_path = os.path.join(input_dir, f)
        parser = TeslaReportParser(file_path)
        all_results.extend(parser.parse())

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Indexed {len(all_results)} pages to {output_file}")


if __name__ == "__main__":
    BASE_PATH = r"D:\tesla_interview"
    IN_DIR = os.path.join(BASE_PATH, "财报")
    OUT_FILE = os.path.join(BASE_PATH, "parsed_tesla_reports.json")
    batch_process(IN_DIR, OUT_FILE)