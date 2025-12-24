import sqlite3
from datetime import datetime
from passlib.context import CryptContext

DB_NAME = "analysis_history.db"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. 사용자 테이블 (아이디, 비번)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    ''')
    
    # 2. 이력 테이블 (user_id 추가됨!)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,  -- [추가] 누가 분석했는지 저장
            filename TEXT NOT NULL,
            analysis_result TEXT NOT NULL,
            risk_level TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

# 3. 사용자 관련 함수
def create_user(username, password):
    """회원가입: 비밀번호를 암호화해서 저장"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    hashed_pw = pwd_context.hash(password) # 암호화
    try:
        cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError: # 이미 있는 아이디일 경우
        return False
    finally:
        conn.close()

def get_user(username):
    """로그인: 아이디로 사용자 정보 찾기"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def verify_password(plain_password, hashed_password):
    """비밀번호 검증"""
    return pwd_context.verify(plain_password, hashed_password)

# 4. 이력 관련 함수 (user_id 추가)
def save_history(user_id, filename, result, risk_level): # user_id 인자 추가
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO analysis_history (user_id, filename, analysis_result, risk_level, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, filename, result, risk_level, now))
    conn.commit()
    conn.close()

def fetch_history(user_id): # user_id 인자 추가
    """내(user_id) 이력만 가져오기"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE user_id = ? ORDER BY id DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "filename": row["filename"],
            "risk_level": row["risk_level"],
            "created_at": row["created_at"],
            "analysis_result": row["analysis_result"]
        })
    return result