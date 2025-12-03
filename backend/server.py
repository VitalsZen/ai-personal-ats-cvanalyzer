# backend/server.py
import os
import shutil
import uvicorn
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from core_logic import analyze_cv_logic
from database import create_db_and_tables, get_session
# Import models mới với cấu trúc JSONB
from models import JobDescription, Application, User, JobDescriptionUpdate

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# LIFESPAN: Quản lý vòng đời DB
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" [DBA] Server init...")
    try:
        # Thử tạo bảng. Nếu DB chưa connect được thì bỏ qua để Server vẫn lên (tránh lỗi No Open Ports)
        create_db_and_tables()
        print(" [DBA] Database schema verified.")
    except Exception as e:
        print(f"[DBA] Database connection warning: {e}")
    yield
    print(" Server shutting down...")

app = FastAPI(title="CareerFlow Enterprise API", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CareerFlow Database System is Operational"}

# SECURITY: QUẢN LÝ PHIÊN NGƯỜI DÙNG (Lazy Registration)
async def get_current_user(
    x_session_id: str = Header(...), 
    session: Session = Depends(get_session)
) -> User:
    """
    Hàm này đóng vai trò như 'Cổng Hải Quan'.
    1. Kiểm tra session_id gửi lên có hợp lệ UUID không.
    2. Tìm trong DB xem User này tồn tại chưa.
    3. Nếu chưa -> Tự động cấp 'Hộ chiếu' (Tạo User mới) ngay lập tức.
    """
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing Session ID")
    
    try:
        user_uuid = uuid.UUID(x_session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Session ID format (Must be UUID)")

    user = session.get(User, user_uuid)
    
    if not user:
        print(f" [DBA] Detected new visitor. Creating Guest User: {user_uuid}")
        user = User(id=user_uuid, is_guest=True)
        session.add(user)
        session.commit()
        session.refresh(user)
        
    return user

# API: JD LIBRARY (Đã áp dụng User Isolation)

@app.get("/api/jds", response_model=List[JobDescription])
def read_jds(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    # CHỈ trả về dữ liệu của chính user đó (Isolation)
    query = select(JobDescription).where(JobDescription.user_id == current_user.id).order_by(JobDescription.created_at.desc())
    return session.exec(query).all()

@app.post("/api/jds", response_model=JobDescription)
def create_jd(
    jd: JobDescription, 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    jd.created_at = datetime.now()
    jd.updated_at = datetime.now()
    jd.user_id = current_user.id # Gắn chủ sở hữu
    
    session.add(jd)
    session.commit()
    session.refresh(jd)
    return jd

@app.patch("/api/jds/{jd_id}")
def update_jd(
    jd_id: int, 
    payload: JobDescriptionUpdate, # Dùng Pydantic model để validate input
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    # Tìm JD và kiểm tra quyền sở hữu cùng lúc
    db_jd = session.exec(
        select(JobDescription).where(JobDescription.id == jd_id, JobDescription.user_id == current_user.id)
    ).first()
    
    if not db_jd:
        raise HTTPException(status_code=404, detail="JD not found or access denied")
    
    jd_data = payload.model_dump(exclude_unset=True)
    for key, value in jd_data.items():
        setattr(db_jd, key, value)
    
    db_jd.updated_at = datetime.now()
    session.add(db_jd)
    session.commit()
    session.refresh(db_jd)
    return db_jd

@app.delete("/api/jds/{jd_id}")
def delete_jd(
    jd_id: int, 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    db_jd = session.exec(
        select(JobDescription).where(JobDescription.id == jd_id, JobDescription.user_id == current_user.id)
    ).first()
    
    if not db_jd:
        raise HTTPException(status_code=404, detail="JD not found or access denied")
        
    session.delete(db_jd)
    session.commit()
    return {"ok": True}

# API: APPLICATIONS (Kết quả AI)

@app.get("/api/applications", response_model=List[Application])
def read_applications(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    query = select(Application).where(Application.user_id == current_user.id).order_by(Application.created_at.desc())
    return session.exec(query).all()

@app.post("/api/applications", response_model=Application)
def create_application(
    app: Application, 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    app.user_id = current_user.id
    if not app.created_at:
        app.created_at = datetime.now()
    
    # Không cần json.dumps thủ công nữa, SQLModel + JSONB tự lo việc này
    session.add(app)
    session.commit()
    session.refresh(app)
    return app

@app.delete("/api/applications/{app_id}")
def delete_application(
    app_id: int, 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    app = session.exec(
        select(Application).where(Application.id == app_id, Application.user_id == current_user.id)
    ).first()
    
    if not app: 
        raise HTTPException(404, detail="Not found or access denied")
    session.delete(app)
    session.commit()
    return {"ok": True}

@app.patch("/api/applications/{app_id}")
def update_application(
    app_id: int, 
    payload: dict, 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    app = session.exec(
        select(Application).where(Application.id == app_id, Application.user_id == current_user.id)
    ).first()
    
    if not app:
        raise HTTPException(404, detail="Not found or access denied")
        
    for key, value in payload.items():
        if hasattr(app, key):
            setattr(app, key, value)
            
    session.add(app)
    session.commit()
    session.refresh(app)
    return app

# ANALYZE ENDPOINT
@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...), 
    jd_text: Optional[str] = Form(None),
    jd_id: Optional[int] = Form(None),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user) # Bắt buộc phải có User mới cho phân tích
):
    final_jd_text = ""
    if jd_id:
        # Lấy JD từ DB, đảm bảo đúng chủ sở hữu
        jd_record = session.exec(
            select(JobDescription).where(JobDescription.id == jd_id, JobDescription.user_id == current_user.id)
        ).first()
        
        if not jd_record: 
            raise HTTPException(404, detail="JD ID not found in your library")
        final_jd_text = jd_record.content
    elif jd_text:
        final_jd_text = jd_text
    else:
        raise HTTPException(400, detail="Must provide jd_text OR jd_id")

    temp_path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Gọi Core Logic (đã chuyển sang HuggingFace + Gemini 1.5 Flash)
        result = analyze_cv_logic(temp_path, final_jd_text)
        
        if "error" in result: raise HTTPException(500, detail=result["error"])
        return result
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
