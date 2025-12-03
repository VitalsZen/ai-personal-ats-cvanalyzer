# backend/core_logic.py
import os
import time
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Union

# --- Imports ---
import pdfplumber 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import HarmBlockThreshold, HarmCategory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

load_dotenv()

# --- DATA MODELS ---
class JobMatchResult(BaseModel):
    personal_info: Dict[str, str] = Field(description="Name, position, experience extracted from CV")
    matching_score: Dict[str, Union[int, str]] = Field(description="Percentage score and explanation")
    requirements_breakdown: Dict[str, str] = Field(description="Ratios for must-have and nice-to-have criteria")
    matched_keywords: List[str] = Field(description="List of matching technical skills")
    radar_chart: Dict[str, int] = Field(description="Scores 1-10 for 5 dimensions")
    radar_reasoning: Dict[str, str] = Field(description="Explanation for each radar score in Vietnamese")
    bilingual_content: Dict[str, Union[Dict, List]] = Field(description="Assessment content in EN and VI")
    
CORE_PROMPT = """
Bạn là một Trợ lý Tuyển dụng AI chuyên nghiệp (JobMatchr). Nhiệm vụ của bạn là phân tích CV (được cung cấp dưới dạng text) và Mô tả công việc (JD - mỗi dòng là một yêu cầu).

**INPUT DATA:**
1. CV Text: {cv_text}
2. JD Text: {jd_text} (Lưu ý: Mỗi dòng trong JD là một tiêu chí riêng biệt).

**NHIỆM VỤ:**
Hãy thực hiện các bước sau một cách logic:

BƯỚC 1: TRÍCH XUẤT THÔNG TIN CÁ NHÂN
- Tìm Name, Position (Vị trí ứng tuyển/hiện tại), Experience (Tổng số năm kinh nghiệm - chỉ lấy số).

BƯỚC 2: PHÂN TÍCH JD VÀ TÍNH ĐIỂM (QUY TẮC "1 ĐỀU")
- Tách JD thành các dòng riêng biệt. Tổng số dòng = Tổng yêu cầu (Total_Req).
- Phân loại từng dòng thành "Bắt buộc" (Requirement) hoặc "Ưu tiên" (Nice-to-have) dựa trên từ khóa:
  * Tiếng Anh: "nice to have", "plus", "preferred", "advantage", "desired", "bonus", "optional", "willing to".
  * Tiếng Việt: "ưu tiên", "lợi thế", "điểm cộng", "không bắt buộc", "mong muốn", "nếu có".
  -> Các trường hợp còn lại mặc định là **"Bắt buộc" (Requirement)**.
- Đối chiếu CV: Với mỗi dòng JD, nếu CV có bằng chứng đáp ứng => Tính là 1 điểm (Matched).
- Keyword phát hiện: Trích xuất các từ khóa kỹ thuật (Hard skill) trùng khớp giữa CV và JD.
- Công thức tính % chung: (Tổng số dòng Matched / Tổng số dòng JD) * 100.

BƯỚC 3: ĐÁNH GIÁ SONG NGỮ (ANH & VIỆT)
- Tạo nội dung đánh giá cho các mục: Đánh giá chung, Điểm mạnh, Điểm yếu (Missing skills), Câu hỏi phỏng vấn.
- Nội dung Tiếng Anh viết trước, Tiếng Việt dịch sát nghĩa theo sau.

BƯỚC 4: CHẤM ĐIỂM RADAR (1-10) VÀ GIẢI THÍCH LÝ DO
*Bắt buộc chấm dựa trên Barem sau:*
1. **Hard Skills (Kỹ năng cứng):** 1-4 (Thiếu nhiều), 5-7 (Cơ bản), 8-10 (Đầy đủ/Nâng cao).
2. **Soft Skills (Kỹ năng mềm):** 1-4 (Sơ sài), 5-7 (Có nhắc đến), 8-10 (Có ví dụ cụ thể).
3. **Experience (Kinh nghiệm):** 1-4 (Ít/Trái ngành), 5-7 (Tương đối), 8-10 (Vượt yêu cầu).
4. **Education (Học vấn):** 1-4 (Không liên quan), 5-7 (Đúng ngành), 8-10 (Bằng cấp cao/Chứng chỉ xịn).
5. **Domain Knowledge (Hiểu biết ngành):** 1-4 (Chung chung), 5-7 (Hiểu quy trình), 8-10 (Am hiểu nghiệp vụ sâu).

**OUTPUT FORMAT (BẮT BUỘC JSON):**
Chỉ trả về 1 JSON duy nhất.
LƯU Ý QUAN TRỌNG:
1. Không dùng Markdown (```json ... ```). Trả về raw text.
2. KHÔNG ĐƯỢC có dấu phẩy (,) ở cuối danh sách hoặc object cuối cùng. (NO TRAILING COMMAS).
3. Đảm bảo cấu trúc ngoặc {{}} đóng mở chính xác. 

Cấu trúc như sau:
{{
    "personal_info": {{
        "name": "String",
        "position": "String (Single title only, e.g., 'Backend Developer')",
        "experience": "String (Single value only, e.g., '2 years')"
    }},
    "matching_score": {{
        "percentage": Integer,
        "explanation": "String (e.g., 'Matched 8/10 requirements')"
    }},
    "requirements_breakdown": {{
        "must_have_ratio": "String (e.g., '5/7')",
        "nice_to_have_ratio": "String (e.g., '3/3')"
    }},
    "matched_keywords": ["String", "String", ...],
    "radar_chart": {{
        "Hard Skills": Integer,
        "Soft Skills": Integer,
        "Experience": Integer,
        "Education": Integer,
        "Domain Knowledge": Integer
    }},
    "radar_reasoning": {{
        "Hard Skills": {{ "en": "English explanation...", "vi": "Giải thích tiếng Việt..." }},
        "Soft Skills": {{ "en": "...", "vi": "..." }},
        "Experience": {{ "en": "...", "vi": "..." }},
        "Education": {{ "en": "...", "vi": "..." }},
        "Domain Knowledge": {{ "en": "...", "vi": "..." }}
    }},
    "bilingual_content": {{
        "general_assessment": {{
            "en": "String",
            "vi": "String"
        }},
        "comparison_table": [
            {{
                "jd_requirement": "String (Original JD line)",
                "cv_evidence": "String (Evidence from CV or 'Not found')",
                "status": "Matched/Not Matched"
            }}
        ],
        "strengths": {{
            "en": ["String", "String"],
            "vi": ["String", "String"]
        }},
        "weaknesses_missing_skills": {{
            "en": ["String", "String"],
            "vi": ["String", "String"]
        }},
        "interview_questions": {{
            "en": ["String", "String"],
            "vi": ["String", "String"]
        }}
    }}
}}
"""

_llm_instance = None
_embedding_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: print("LỖI: GOOGLE_API_KEY chưa cấu hình!")
        
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0.2,
            google_api_key=api_key,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    return _llm_instance

def get_embeddings():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embedding_instance

# Hàm vệ sinh text (Fix lỗi ngoặc nhọn trong CV code/css)
def sanitize_text_for_prompt(text):
    if not text: return ""
    return text.replace("{", "(").replace("}", ")")

def clean_json_string(json_str):
    try:
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx : end_idx + 1]
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        return json_str
    except Exception:
        return json_str 

def analyze_cv_logic(file_path: str, jd_text: str):
    if not os.getenv("GOOGLE_API_KEY"):
        return {"error": "Server Config Error: Missing GOOGLE_API_KEY"}

    # 1. Đọc PDF
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                full_text += (page.extract_text() or "") + "\n"
        
        if not full_text.strip(): return {"error": "Empty PDF content"}
        
        # [FIX] Vệ sinh văn bản
        full_text = sanitize_text_for_prompt(full_text)
        jd_text = sanitize_text_for_prompt(jd_text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([full_text])
    except Exception as e:
        return {"error": f"PDF Error: {str(e)}"}

    # 2. RAG Logic
    try:
        embeddings = get_embeddings()
        llm = get_llm()

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=f"cv_{int(time.time())}",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # [FIX QUAN TRỌNG] Thay đổi phương thức gọi retriever
        # Trong bản LangChain mới, dùng .invoke() thay vì .get_relevant_documents()
        relevant_docs = retriever.invoke(jd_text)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])

        # Gọi LLM trực tiếp (Tránh lỗi Chain missing variables)
        prompt = ChatPromptTemplate.from_template(CORE_PROMPT)
        final_prompt_value = prompt.format_messages(
            cv_text=context_text, 
            jd_text=jd_text
        )
        
        print(" Analyzing with Gemini 1.5 Flash...")
        response = llm.invoke(final_prompt_value)
        
        # Xử lý kết quả
        cleaned_content = clean_json_string(response.content)
        
        try:
            result = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            print(f" JSON Error: {cleaned_content[:100]}...")
            return {"error": "AI returned invalid JSON. Please try again."}
        
        vectorstore.delete_collection() 
        return result

    except Exception as e:
        print(f" System Error: {str(e)}")
        return {"error": f"System Error: {str(e)}"}
