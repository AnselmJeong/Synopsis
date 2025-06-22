"""
환경 설정 및 구성 관리
"""

import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class Config:
    """애플리케이션 설정 클래스"""

    # Gemini API 설정
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # 데이터베이스 설정
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://user@localhost:5432/postgres"
    )

    # 새로운 청킹 시스템 설정
    TARGET_CHUNK_SIZE = int(os.getenv("TARGET_CHUNK_SIZE", "4096"))
    MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "1000"))
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "6000"))

    # 레거시 청킹 설정 (하위 호환성)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", str(TARGET_CHUNK_SIZE)))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # 로깅 설정
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # 검색 설정
    DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))

    @property
    def gemini_api_key(self):
        """Gemini API 키"""
        return self.GEMINI_API_KEY

    @property
    def database_url(self):
        """데이터베이스 URL"""
        return self.DATABASE_URL

    @property
    def target_chunk_size(self):
        """목표 청크 크기"""
        return self.TARGET_CHUNK_SIZE

    @property
    def min_chunk_size(self):
        """최소 청크 크기"""
        return self.MIN_CHUNK_SIZE

    @property
    def max_chunk_size(self):
        """최대 청크 크기"""
        return self.MAX_CHUNK_SIZE

    @classmethod
    def validate(cls):
        """설정 유효성 검사"""
        errors = []

        if not cls.DATABASE_URL:
            errors.append("DATABASE_URL이 설정되지 않았습니다.")

        # 새로운 청킹 시스템 검증
        if cls.TARGET_CHUNK_SIZE <= 0:
            errors.append("TARGET_CHUNK_SIZE는 0보다 커야 합니다.")

        if cls.MIN_CHUNK_SIZE <= 0:
            errors.append("MIN_CHUNK_SIZE는 0보다 커야 합니다.")

        if cls.MAX_CHUNK_SIZE <= 0:
            errors.append("MAX_CHUNK_SIZE는 0보다 커야 합니다.")

        if cls.MIN_CHUNK_SIZE >= cls.MAX_CHUNK_SIZE:
            errors.append("MIN_CHUNK_SIZE는 MAX_CHUNK_SIZE보다 작아야 합니다.")

        if cls.TARGET_CHUNK_SIZE < cls.MIN_CHUNK_SIZE:
            errors.append(
                "TARGET_CHUNK_SIZE는 MIN_CHUNK_SIZE보다 크거나 같아야 합니다."
            )

        if cls.TARGET_CHUNK_SIZE > cls.MAX_CHUNK_SIZE:
            errors.append(
                "TARGET_CHUNK_SIZE는 MAX_CHUNK_SIZE보다 작거나 같아야 합니다."
            )

        # 레거시 청킹 시스템 검증 (하위 호환성)
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE는 0보다 커야 합니다.")

        if cls.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP은 0 이상이어야 합니다.")

        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP은 CHUNK_SIZE보다 작아야 합니다.")

        # 검색 설정 검증
        if not (0 <= cls.SIMILARITY_THRESHOLD <= 1):
            errors.append("SIMILARITY_THRESHOLD는 0과 1 사이의 값이어야 합니다.")

        if not (0 <= cls.SEMANTIC_WEIGHT <= 1):
            errors.append("SEMANTIC_WEIGHT는 0과 1 사이의 값이어야 합니다.")

        if not (0 <= cls.KEYWORD_WEIGHT <= 1):
            errors.append("KEYWORD_WEIGHT는 0과 1 사이의 값이어야 합니다.")

        if abs(cls.SEMANTIC_WEIGHT + cls.KEYWORD_WEIGHT - 1.0) > 0.001:
            errors.append("SEMANTIC_WEIGHT + KEYWORD_WEIGHT는 1.0이어야 합니다.")

        return errors

    @classmethod
    def print_config(cls):
        """현재 설정을 출력합니다."""
        print("현재 설정:")
        print(f"  데이터베이스 URL: {cls.DATABASE_URL}")
        print(f"  목표 청크 크기: {cls.TARGET_CHUNK_SIZE}")
        print(f"  최소 청크 크기: {cls.MIN_CHUNK_SIZE}")
        print(f"  최대 청크 크기: {cls.MAX_CHUNK_SIZE}")
        print(f"  레거시 청크 크기: {cls.CHUNK_SIZE}")
        print(f"  레거시 청크 중복: {cls.CHUNK_OVERLAP}")
        print(f"  로그 레벨: {cls.LOG_LEVEL}")
        print(f"  기본 검색 제한: {cls.DEFAULT_SEARCH_LIMIT}")
        print(f"  유사도 임계값: {cls.SIMILARITY_THRESHOLD}")
        print(f"  의미적 검색 가중치: {cls.SEMANTIC_WEIGHT}")
        print(f"  키워드 검색 가중치: {cls.KEYWORD_WEIGHT}")
        print(f"  Gemini API 키 설정됨: {'예' if cls.GEMINI_API_KEY else '아니오'}")


def validate_config(config: Config) -> None:
    """
    설정 유효성 검사 함수

    Args:
        config: Config 인스턴스

    Raises:
        ValueError: 설정이 유효하지 않은 경우
    """
    errors = config.validate()
    if errors:
        error_message = "설정 오류가 발견되었습니다:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_message)


# 설정 유효성 검사 수행 (모듈 로드 시)
try:
    config_instance = Config()
    validate_config(config_instance)
except ValueError as e:
    print(f"⚠️ {e}")
    print("설정을 확인하고 다시 시도하세요.")
