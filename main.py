from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
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
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")

doc_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT, 
    credential=AzureKeyCredential(DOC_INTEL_KEY)
)

openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=OPENAI_ENDPOINT
)

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

@app.post("/analyze-contract")
async def analyze_contract(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user) # 로그인 체크
):
    try:
        # 1.OCR 텍스트 추출 및 전처리
        content = await file.read()
        poller = doc_client.begin_analyze_document("prebuilt-read", body=content)
        result = poller.result()
        raw_text = "\n".join([p.content for p in result.paragraphs])
        
        # 전처리 함수 적용 (깨진 글자 청소)
        contract_text = clean_ocr_text(raw_text)

        # 2.AI 분석 (Search 연동)
        system_message = """
        당신은 대한민국 '주택임대차보호법' 및 관련 판례 데이터를 기반으로 부동산 계약서를 검토하는 법률 AI 어시스턴트입니다.
        사용자가 업로드한 [계약서]의 내용을 [검색된 법률 조항]과 **조항 대 조항(Clause-by-Clause)으로 대조**하여 분석하세요.
        
        [분석 지침]
        1. **위법성 검토:** 계약서의 내용 중 '주택임대차보호법'의 강행규정에 위반되어 임차인에게 불리한 조항이 있는지 찾으세요.
        2. **독소 조항 탐지:** 임차인에게 과도한 의무를 부과하거나 임대인의 의무를 회피하는 내용이 있는지 확인하세요.
        3. **필수 권리 누락 확인:** 우선변제권, 대항력 확보 등 필수 보호 장치가 누락되었는지 확인하세요.
        
        [출처 표기 원칙]
        - 반드시 근거 법령명과 조항 번호를 명시하세요. (예: [doc1])
        """

        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"이 계약서를 분석해줘:\n{contract_text}"}
            ],
            extra_body={
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": SEARCH_ENDPOINT,
                            "index_name": SEARCH_INDEX_NAME,
                            "authentication": {
                                "type": "api_key",
                                "key": SEARCH_KEY
                            },
                            "top_n_documents": 5,
                            "in_scope": True,
                            "strictness": 4,
                            "role_information": system_message
                        }
                    }
                ]
            }
        )

        raw_analysis = response.choices[0].message.content
        
        # 후처리 함수 적용 (마크다운 기호 제거)
        analysis_result = clean_ai_response(raw_analysis)
        database.save_history(
            current_user['id'],
            file.filename,       
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