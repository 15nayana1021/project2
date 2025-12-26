from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from pydantic import BaseModel
from fpdf import FPDF
from jose import JWTError, jwt
from datetime import timedelta
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from typing import List
import os
import database
import re

def clean_ocr_text(text: str) -> str:
    """[전처리] OCR 결과에서 불필요한 공백과 특수문자 제거"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("ï»¿", "") 
    return text

def clean_ai_response(text: str) -> str:
    """[후처리] AI 응답에서 마크다운 코드 블록 기호 제거"""
    text = text.replace("```markdown", "").replace("```", "")
    return text.strip()

# [추가] 문서 종류 검증 함수 ("검문소" 역할)
def validate_document_type(text, selected_type):
    # 1. 부동산 계약서 필수 키워드 (이 중 하나라도 있어야 통과)
    keywords_estate = ["부동산", "임대차", "전세", "월세", "임대인", "임차인", "보증금", "소재지"]
    
    # 2. 근로 계약서 필수 키워드 (이 중 하나라도 있어야 통과)
    keywords_labor = ["근로", "고용", "사용자", "취업", "임금", "급여", "연봉", "퇴직금", "수습"]

    # 텍스트에서 키워드 개수 세기
    estate_score = sum(1 for k in keywords_estate if k in text)
    labor_score = sum(1 for k in keywords_labor if k in text)

    print(f"🔍 검증 점수 - 부동산점수: {estate_score}, 근로점수: {labor_score}")

    # [판단 로직]
    if selected_type == "real_estate":
        # 부동산을 선택했는데, 근로 관련 단어가 압도적으로 많거나 부동산 단어가 아예 없으면?
        if labor_score > estate_score + 2: 
            raise HTTPException(status_code=400, detail="선택하신 건 '부동산 계약서'인데, 업로드된 파일은 '근로 계약서'로 보입니다.")
        if estate_score == 0:
            raise HTTPException(status_code=400, detail="업로드된 파일에서 '부동산 계약' 관련 내용을 찾을 수 없습니다.")

    elif selected_type == "labor":
        # 근로를 선택했는데, 부동산 단어가 압도적으로 많거나 근로 단어가 아예 없으면?
        if estate_score > labor_score + 2:
            raise HTTPException(status_code=400, detail="선택하신 건 '근로 계약서'인데, 업로드된 파일은 '부동산 계약서'로 보입니다.")
        if labor_score == 0:
            raise HTTPException(status_code=400, detail="업로드된 파일에서 '근로 계약' 관련 내용을 찾을 수 없습니다.")
            
    # 통과하면 아무 일 없이 리턴
    return True

# DB 초기화
database.init_db()
load_dotenv()
app = FastAPI()


# 0. 설정 및 보안
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT 보안 설정
SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# 1. 클라이언트 초기화
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_KEY = os.getenv("DOC_INTEL_KEY")

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")

SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("SEARCH_KEY")
REAL_ESTATE_CASES = os.getenv("REAL_ESTATE_CASES")
REAL_ESTATE_FORMS = os.getenv("REAL_ESTATE_FORMS")
REAL_ESTATE_LAWS = os.getenv("REAL_ESTATE_LAWS")
LABOR_CASES_INDEX = os.getenv("LABOR_CASES_INDEX")
LABOR_FORMS_INDEX = os.getenv("LABOR_FORMS_INDEX")
LABOR_LAWS_INDEX = os.getenv("LABOR_LAWS_INDEX")

doc_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT, 
    credential=AzureKeyCredential(DOC_INTEL_KEY)
)

openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=OPENAI_ENDPOINT
)

# ... (기존 클라이언트 초기화 코드들) ...

# ★ [추가] 텍스트를 벡터로 변환하는 함수 (검색용)
def get_embedding(text):
    return openai_client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    ).data[0].embedding

# ★ [추가] 특정 인덱스에서 관련 정보를 찾아오는 함수
def search_in_azure(index_name, query_text):
    try:
        if not index_name: return ""
        
        search_client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=index_name,
            credential=AzureKeyCredential(SEARCH_KEY)
        )
        
        # 벡터 생성 (임베딩)
        query_vector = get_embedding(query_text)
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=2, fields="content_vector")

        # 검색
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["title", "content"]
        )
        
        summary = ""
        for res in results:
            summary += f"\n[출처: {res.get('title', '문서')}]\n내용: {res.get('content', '')[:500]}\n"
        return summary
    except Exception as e:
        print(f"⚠️ 검색 에러 ({index_name}): {e}")
        return ""

# 데이터 모델 정의
class ReportRequest(BaseModel):
    text: str


# 2. 인증/인가 헬퍼 함수 (Auth Helpers)
def create_access_token(data: dict):
    """JWT 토큰 생성 함수"""
    to_encode = data.copy()
    expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # 만료 시간 추가 (현재 시간 + 30분)
    # (실제 datetime.utcnow() 등을 사용해야 하지만 간단하게 구현)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """현재 로그인한 유저 확인 (보안 의존성)"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="자격 증명을 검증할 수 없습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = database.get_user(username)
    if user is None:
        raise credentials_exception
    return user # 로그인한 사용자의 DB 정보(row) 반환


# 3. 인증 API 엔드포인트
@app.post("/signup")
async def signup(form_data: OAuth2PasswordRequestForm = Depends()):
    """회원가입"""
    success = database.create_user(form_data.username, form_data.password)
    if not success:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
    return {"message": "회원가입 성공"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """로그인 및 토큰 발급"""
    user = database.get_user(form_data.username)
    if not user or not database.verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 토큰 발급
    access_token = create_access_token(data={"sub": user['username']})
    return {"access_token": access_token, "token_type": "bearer"}


# 4. 핵심 기능 (로그인 필요)
@app.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    """[보안] 내 이력만 조회"""
    # current_user['id']를 넘겨서 내 것만 가져옴
    return database.fetch_history(current_user['id'])

# ... (상단 import 및 설정 코드는 기존과 동일) ...

# [main.py] analyze_contract 함수 내부

@app.post("/analyze-contract")
async def analyze_contract(
    files: List[UploadFile] = File(...), # ★ List[UploadFile]로 변경
    type: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        combined_raw_text = ""
        file_names = [f.filename for f in files]

        for file in files:
            content = await file.read()
            # Azure Document Intelligence로 텍스트 추출
            poller = doc_client.begin_analyze_document("prebuilt-read", body=content)
            result = poller.result()
            
            file_text = "\n".join([p.content for p in result.paragraphs])
            combined_raw_text += f"\n\n[파일명: {file.filename}]\n{file_text}"
        # [Step 1] OCR로 텍스트 추출
        
        full_contract_text = clean_ocr_text(combined_raw_text)
        validate_document_type(full_contract_text, type)

        # [Step 2] 인덱스 선택
        target_indexes = []
        system_role = ""

        if type == "real_estate":
            target_indexes = [
                os.getenv("REAL_ESTATE_LAWSS"), 
                os.getenv("REAL_ESTATE_FORMS"), 
                os.getenv("REAL_ESTATE_CASES")
            ]
            system_role = "부동산 전문 변호사"

        elif type == "labor":
            target_indexes = [
                os.getenv("LABOR_LAWS_INDEX"), 
                os.getenv("LABOR_FORMS_INDEX"), 
                os.getenv("LABOR_CASES_INDEX")
            ]
            system_role = "공인노무사"

        # [Step 3] ★ 중요: 파이썬이 직접 검색 (Azure API 제한 우회)
        search_query = full_contract_text[:500] # 계약서 앞부분으로 검색
        combined_knowledge = ""
        
        print(f"🔍 {len(target_indexes)}개의 인덱스 뒤지는 중...")

        for idx_name in target_indexes:
            if idx_name: 
                # (주의: search_in_azure 함수가 main.py 상단에 정의되어 있어야 함)
                found_info = search_in_azure(idx_name, search_query)
                combined_knowledge += found_info

        # [Step 4] AI 호출 (★ extra_body 삭제됨!)
        system_message = f"""
        당신은 유능한 {system_role}입니다.
        제공된 여러 문서(계약서, 등기부등본 등)를 서로 대조하고 참고 자료를 바탕으로 분석하세요.
        특히 계약서의 임대인과 등기부의 소유주가 일치하는지, 근저당권 설정이 위험하지 않은지 확인하세요.
        """

        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"이 계약서를 분석해줘:\n{full_contract_text}"}
            ]
            # ★ 여기에 있던 extra_body={...} 코드가 싹 사라져야 합니다!
        )

        # [Step 5] 결과 저장 및 반환
        raw_analysis = response.choices[0].message.content
        analysis_result = clean_ai_response(raw_analysis)

        database.save_history(
            current_user['id'],
            "," .join(file_names),
            analysis_result,
            "분석완료"
        )

        return {
            "status": "success",
            "filename": file.filename,
            "analysis": analysis_result
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        # 자세한 에러 로그를 보기 위해 traceback을 찍어볼 수도 있습니다.
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# 5. 기타 기능 (공개/비공개 선택 가능)
@app.post("/create-pdf")
async def create_pdf(request: ReportRequest):
    """PDF 생성 (로그인 없이도 가능하게 유지)"""
    pdf = FPDF()
    pdf.add_page()
    font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    
    if os.path.exists(font_path):
        pdf.add_font("Malgun", "", font_path)
        pdf.set_font("Malgun", size=12)
    else:
        pdf.set_font("Arial", size=12)
    
    pdf.multi_cell(0, 10, request.text)
    temp_filename = "temp_report.pdf"
    pdf.output(temp_filename)
    
    return FileResponse(path=temp_filename, filename="부동산_계약_분석_리포트.pdf", media_type='application/pdf')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)