"""
PostgreSQL 데이터베이스 스키마 설정 모듈
pgvector와 tsvector 확장 기능을 활성화하고 documents 테이블을 생성합니다.
"""

import psycopg
from psycopg import sql
import logging

# 로깅 설정
logger = logging.getLogger(__name__)


def setup_database(connection_string="postgresql://user@localhost:5432/postgres"):
    """
    데이터베이스 스키마를 설정합니다.

    Args:
        connection_string (str): PostgreSQL 연결 문자열
    """
    try:
        # 데이터베이스 연결
        conn = psycopg.connect(connection_string)
        conn.autocommit = True
        cursor = conn.cursor()

        logger.info("데이터베이스에 연결되었습니다.")

        # 필요한 확장 기능 활성화
        logger.info("필요한 확장 기능들을 활성화하는 중...")

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector 확장 기능이 활성화되었습니다.")

        # documents 테이블 생성
        logger.info("documents 테이블을 생성하는 중...")

        create_table_query = """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(768),
            ts_content TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
            hierarchy JSONB,
            metadata JSONB,
            book_title TEXT,
            book_author TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor.execute(create_table_query)
        logger.info("documents 테이블이 생성되었습니다.")

        # 인덱스 생성
        logger.info("인덱스들을 생성하는 중...")

        # 전문검색을 위한 GIN 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_fts 
            ON documents USING GIN (ts_content);
        """)
        logger.info("전문검색 GIN 인덱스가 생성되었습니다.")

        # 벡터 유사성 검색을 위한 HNSW 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding 
            ON documents USING hnsw (embedding vector_cosine_ops) 
            WITH (m=16, ef_construction=64);
        """)
        logger.info("벡터 유사성 검색 HNSW 인덱스가 생성되었습니다.")

        # 계층 구조 검색을 위한 GIN 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_hierarchy 
            ON documents USING GIN (hierarchy);
        """)
        logger.info("계층 구조 GIN 인덱스가 생성되었습니다.")

        # 메타데이터 검색을 위한 GIN 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_metadata 
            ON documents USING GIN (metadata);
        """)
        logger.info("메타데이터 GIN 인덱스가 생성되었습니다.")

        # 업데이트 트리거 함수 생성
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
               NEW.updated_at = CURRENT_TIMESTAMP;
               RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)

        # 트리거가 이미 존재하는지 확인
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_trigger 
                WHERE tgname = 'update_documents_updated_at'
            );
        """)

        trigger_exists = cursor.fetchone()[0]
        if not trigger_exists:
            cursor.execute("""
                CREATE TRIGGER update_documents_updated_at 
                BEFORE UPDATE ON documents 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            """)
            logger.info("업데이트 트리거가 생성되었습니다.")
        else:
            logger.info("업데이트 트리거가 이미 존재합니다.")

        logger.info("데이터베이스 스키마 설정이 완료되었습니다.")

        # 연결 종료
        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"데이터베이스 설정 중 오류가 발생했습니다: {e}")
        raise


def check_database_status(
    connection_string="postgresql://user@localhost:5432/postgres",
):
    """
    데이터베이스 상태를 확인합니다.

    Args:
        connection_string (str): PostgreSQL 연결 문자열
    """
    try:
        conn = psycopg.connect(connection_string)
        cursor = conn.cursor()

        # 확장 기능 확인
        cursor.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector');")
        extensions = cursor.fetchall()
        logger.info(f"설치된 확장 기능: {[ext[0] for ext in extensions]}")

        # 테이블 존재 확인
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'documents';
        """)
        tables = cursor.fetchall()
        logger.info(f"존재하는 테이블: {[table[0] for table in tables]}")

        # 인덱스 확인
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'documents';
        """)
        indexes = cursor.fetchall()
        logger.info(f"documents 테이블의 인덱스: {[idx[0] for idx in indexes]}")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"데이터베이스 상태 확인 중 오류가 발생했습니다: {e}")
        raise
