# Synopsis - EPUB 분석 및 검색 시스템

스마트한 EPUB 콘텐츠 분석과 시맨틱 검색을 제공하는 Python 패키지입니다. 복잡한 전자책을 자연스러운 소제목 단위로 청킹하여 벡터 데이터베이스에 저장하고, AI 기반 검색 기능을 제공합니다.

## 주요 기능

- **🔍 스마트 청킹**: 자연스러운 소제목을 자동 감지하여 의미 있는 단위로 콘텐츠 분할
- **🎯 하이브리드 검색**: 시맨틱 검색과 키워드 검색을 결합한 정확한 검색
- **📊 벡터 데이터베이스**: PostgreSQL + pgvector를 활용한 고성능 임베딩 저장소
- **🖥️ CLI 도구**: 명령행에서 바로 사용 가능한 인터페이스
- **📖 대화형 뷰어**: 검색 결과를 직관적으로 탐색할 수 있는 인터페이스

## 로컬 설치 및 설정

### 1. 개발 모드로 설치 (추천)

```bash
# 저장소 클론
git clone <repository-url>
cd synopsis

# 가상환경 생성 (선택사항이지만 권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 개발 모드로 설치 (편집 가능한 설치)
pip install -e .
```

개발 모드 설치의 장점:
- 코드 수정 시 재설치 없이 바로 반영
- 로컬에서 지속적으로 개발 및 테스트 가능
- PyPI 없이도 다른 프로젝트에서 import 가능

### 2. 직접 설치

```bash
# 현재 디렉토리에서 설치
pip install .

# 또는 wheel 빌드 후 설치
pip install build
python -m build
pip install dist/synopsis-0.1.0-py3-none-any.whl
```

### 3. PYTHONPATH 사용 (임시 방법)

```bash
# 환경 변수로 패키지 경로 추가
export PYTHONPATH=$PYTHONPATH:/path/to/synopsis

# 스크립트 실행
python your_script.py
```

### 4. 환경 변수 설정

`.env` 파일을 생성하여 필요한 환경 변수를 설정하세요:

```bash
# .env 파일 예제
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://username:password@localhost:5432/synopsis
```

### 5. 데이터베이스 설정

PostgreSQL에 pgvector 확장을 설치하고 데이터베이스를 준비하세요:

```sql
-- PostgreSQL에서 실행
CREATE DATABASE synopsis;
\c synopsis;
CREATE EXTENSION vector;
```

## 다른 프로젝트에서 사용하기

### 기본 사용법

개발 모드로 설치한 후 다른 Python 프로젝트에서 바로 import할 수 있습니다:

```python
# 다른 프로젝트의 Python 파일에서
from synopsis import EPUBProcessor, SearchSystem
from synopsis.models import ContentChunk
from synopsis.utils.format import interactive_result_viewer

# 모든 기능을 정상적으로 사용 가능
processor = EPUBProcessor(...)
search_system = SearchSystem(...)
```

### 프로젝트별 설치 방법

1. **Git 저장소에서 직접 설치**:
   ```bash
   pip install git+https://github.com/your-username/synopsis.git
   ```

2. **로컬 경로에서 설치**:
   ```bash
   pip install /path/to/synopsis
   ```

3. **개발 모드로 설치** (수정사항 자동 반영):
   ```bash
   pip install -e /path/to/synopsis
   ```

## 패키지 공유 방법 (PyPI 없이)

### 1. 압축 파일로 공유

```bash
# 압축 파일 생성
tar -czf synopsis.tar.gz synopsis/

# 또는 zip 파일로
zip -r synopsis.zip synopsis/

# 다른 컴퓨터에서 사용
tar -xzf synopsis.tar.gz  # 또는 unzip synopsis.zip
cd synopsis
pip install -e .
```

### 2. Wheel 파일로 배포

```bash
# wheel 파일 생성
pip install build
python -m build

# 생성된 파일들
ls dist/
# synopsis-0.1.0-py3-none-any.whl
# synopsis-0.1.0.tar.gz

# 다른 환경에서 설치
pip install synopsis-0.1.0-py3-none-any.whl
```

### 3. Docker를 활용한 배포

```dockerfile
# Dockerfile 예제
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["synopsis", "--help"]
```

```bash
# Docker 이미지 빌드 및 실행
docker build -t synopsis .
docker run synopsis synopsis search "검색어"
```

## 사용법

### CLI 도구 사용

설치 후 `synopsis` 명령어를 사용할 수 있습니다:

```bash
# 데이터베이스 테이블 생성
synopsis setup

# EPUB 파일 처리 및 청킹
synopsis process "book/your_book.epub"

# 고급 청킹 옵션
synopsis process "book.epub" --target-chunk-size 4096 --min-chunk-size 2048 --max-chunk-size 8192

# 하이브리드 검색 (기본)
synopsis search "인공지능의 발전"

# 시맨틱 검색만 사용
synopsis search "machine learning" --search-type semantic

# 키워드 검색만 사용  
synopsis search "neural network" --search-type keyword

# 상세 결과 보기
synopsis search "quantum physics" --full-content --detailed-scores

# 대화형 결과 뷰어
synopsis search "relativity" --interactive

# 대화형 검색 모드
synopsis interactive
```

### Python 패키지로 사용

```python
from synopsis import EPUBProcessor, SearchSystem, Config

# 설정 로드
config = Config()

# EPUB 처리
processor = EPUBProcessor(
    gemini_api_key=config.gemini_api_key,
    connection_string=config.database_url,
    target_chunk_size=4096
)

# 파일 처리
processor.process_epub("path/to/book.epub")

# 검색 시스템 초기화
search_system = SearchSystem(
    connection_string=config.database_url,
    gemini_api_key=config.gemini_api_key
)

# 검색 실행
from synopsis.core.search import SearchType

results = search_system.search(
    query="양자역학의 기본 원리",
    search_type=SearchType.HYBRID,
    limit=5
)

# 결과 출력
from synopsis.utils.format import format_search_results

formatted = format_search_results(results, show_full_content=True)
print(formatted)
```

### 고급 사용법

```python
# 사용자 정의 청킹 설정
processor = EPUBProcessor(
    gemini_api_key="your-api-key",
    connection_string="postgresql://...",
    target_chunk_size=6000,  # 더 큰 청크
    min_chunk_size=3000,     # 최소 크기 증가
    max_chunk_size=10000,    # 최대 크기 증가
)

# 계층 구조 필터 검색
results = search_system.search(
    query="특정 주제",
    hierarchy_filter={"level": 1}  # 특정 레벨만 검색
)

# 데이터베이스 통계 확인
stats = search_system.get_database_stats()
print(f"총 문서 수: {stats['document_count']}")
print(f"평균 길이: {stats['avg_content_length']}")
```

## 개발 및 테스트

### 개발 환경 설정

```bash
# 개발 의존성 설치
pip install -e ".[dev]"

# 코드 포맷팅
black synopsis/
isort synopsis/

# 타입 체킹 (선택사항)
mypy synopsis/

# 테스트 실행 (테스트가 있는 경우)
pytest
```

### 패키지 검증

```bash
# 설치 확인
pip list | grep synopsis

# import 테스트
python -c "from synopsis import EPUBProcessor, SearchSystem; print('Import successful')"

# CLI 명령어 테스트
synopsis --help
```

## 패키지 구조

```
synopsis/
├── __init__.py          # 메인 패키지 (주요 클래스 export)
├── cli.py              # 명령행 인터페이스
├── core/               # 핵심 기능
│   ├── config.py       # 환경 설정 관리
│   ├── database.py     # 데이터베이스 스키마 설정  
│   ├── processor.py    # EPUB 처리 및 스마트 청킹
│   └── search.py       # 다중 검색 시스템
├── models/             # 데이터 모델
│   ├── toc.py         # TOC 항목 모델
│   └── chunk.py       # 콘텐츠 청크 모델
└── utils/              # 유틸리티
    └── format.py       # 결과 포맷팅 및 뷰어
```

## 의존성

주요 의존성 패키지들:
- `ebooklib`: EPUB 파일 처리
- `psycopg2-binary`: PostgreSQL 연결
- `google-generativeai`: Gemini AI API
- `beautifulsoup4`: HTML 파싱
- `numpy`: 수치 연산
- `python-dotenv`: 환경 변수 관리

## 시스템 요구사항

- Python 3.8+
- PostgreSQL 12+ (pgvector 확장 포함)
- Google Gemini API 키
- 충분한 디스크 공간 (대용량 EPUB 파일 처리 시)

## 문제 해결

### 일반적인 문제들

1. **PostgreSQL 연결 오류**
   ```bash
   # PostgreSQL 서비스 확인
   sudo systemctl status postgresql
   
   # pgvector 확장 설치 확인
   psql -d synopsis -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
   ```

2. **Gemini API 오류**
   ```bash
   # API 키 확인
   echo $GEMINI_API_KEY
   
   # .env 파일 확인
   cat .env | grep GEMINI_API_KEY
   ```

3. **패키지 import 오류**
   ```bash
   # 설치 확인
   pip list | grep synopsis
   
   # 재설치
   pip install -e . --force-reinstall
   ```

4. **권한 오류**
   ```bash
   # 사용자 설치 (권한 문제 시)
   pip install --user -e .
   
   # 또는 가상환경 사용
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

### 디버깅 모드

더 자세한 로그를 보려면 환경 변수를 설정하세요:

```bash
export SYNOPSIS_DEBUG=1
synopsis process book.epub
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 기여하기

1. 저장소를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 향후 계획

- [ ] 더 많은 전자책 포맷 지원 (PDF, MOBI 등)
- [ ] 웹 인터페이스 추가
- [ ] 다국어 콘텐츠 처리 개선
- [ ] 분산 처리 지원
- [ ] 캐싱 시스템 추가
