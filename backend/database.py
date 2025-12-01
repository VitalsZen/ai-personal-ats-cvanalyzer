import os

from sqlmodel import SQLModel, create_engine, Session



# 1. Xác định vị trí lưu file database.db (ngay tại thư mục backend)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sqlite_file_name = os.path.join(BASE_DIR, "database.db")

sqlite_url = f"sqlite:///{sqlite_file_name}"



# 2. Tạo Engine kết nối (check_same_thread=False cần thiết cho SQLite)

connect_args = {"check_same_thread": False}

engine = create_engine(sqlite_url, connect_args=connect_args)



# 3. Hàm tạo bảng (chạy khi server khởi động)

def create_db_and_tables():

    # Nó sẽ tìm tất cả class kế thừa SQLModel và tạo bảng tương ứng

    SQLModel.metadata.create_all(engine)



# 4. Dependency để lấy session làm việc với DB

def get_session():

    with Session(engine) as session:

        yield session