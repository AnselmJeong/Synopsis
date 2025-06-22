"""
다양한 검색 방식을 지원하는 통합 검색 시스템
의미적 검색(Semantic), 키워드 검색(Keyword), 하이브리드 검색을 제공합니다.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)


class SearchType(Enum):
    """검색 유형 열거형"""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """검색 결과를 나타내는 데이터 클래스"""

    id: int
    content: str
    score: float
    hierarchy: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    search_type: str
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    book_title: Optional[str] = None
    book_author: Optional[str] = None


class SearchSystem:
    """통합 검색 시스템 클래스"""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        connection_string: str = "postgresql://user@localhost:5432/postgres",
    ):
        """
        검색 시스템을 초기화합니다.

        Args:
            gemini_api_key: Gemini API 키
            connection_string: PostgreSQL 연결 문자열
        """
        self.gemini_api_key = gemini_api_key
        self.connection_string = connection_string

        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            logger.info("Gemini API 초기화 완료")
        else:
            logger.warning("Gemini API 키가 없어 의미적 검색 기능이 제한됩니다.")

    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        쿼리에 대한 임베딩을 생성합니다.

        Args:
            query: 임베딩을 생성할 쿼리

        Returns:
            임베딩 벡터 또는 None
        """
        if not self.gemini_api_key:
            return None

        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query",
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"쿼리 임베딩 생성 중 오류 발생: {e}")
            return None

    def semantic_search(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        의미적 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            limit: 반환할 결과 수
            similarity_threshold: 유사도 임계값

        Returns:
            검색 결과 리스트
        """
        logger.info(f"의미적 검색 수행: '{query}'")

        # 쿼리 임베딩 생성
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            logger.error("쿼리 임베딩을 생성할 수 없습니다.")
            return []

        try:
            conn = psycopg.connect(self.connection_string)
            register_vector(conn)
            cursor = conn.cursor()

            # 유사도 기반 검색
            cursor.execute(
                """
                SELECT id, content, hierarchy, metadata, book_title, book_author,
                       1 - (embedding <=> %s) as similarity
                FROM documents
                WHERE embedding IS NOT NULL 
                  AND 1 - (embedding <=> %s) > %s
                ORDER BY similarity DESC
                LIMIT %s
            """,
                (query_embedding, query_embedding, similarity_threshold, limit),
            )

            results = []
            for row in cursor.fetchall():
                # JSONB 타입은 이미 파싱되어 딕셔너리로 반환됨
                hierarchy = row[2] if row[2] else []
                metadata = row[3] if row[3] else {}

                result = SearchResult(
                    id=row[0],
                    content=row[1],
                    score=row[6],
                    hierarchy=hierarchy,
                    metadata=metadata,
                    search_type="semantic",
                    semantic_score=row[6],
                    book_title=row[4],
                    book_author=row[5],
                )
                results.append(result)

            logger.info(f"의미적 검색 결과: {len(results)}개")
            return results

        except Exception as e:
            logger.error(f"의미적 검색 중 오류 발생: {e}")
            return []
        finally:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()

    def keyword_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        키워드 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            limit: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        logger.info(f"키워드 검색 수행: '{query}'")

        try:
            conn = psycopg.connect(self.connection_string)
            cursor = conn.cursor()

            # 전문검색 쿼리
            cursor.execute(
                """
                SELECT id, content, hierarchy, metadata, book_title, book_author,
                       ts_rank(ts_content, plainto_tsquery('simple', %s)) as rank
                FROM documents
                WHERE ts_content @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s
            """,
                (query, query, limit),
            )

            results = []
            for row in cursor.fetchall():
                # JSONB 타입은 이미 파싱되어 딕셔너리로 반환됨
                hierarchy = row[2] if row[2] else []
                metadata = row[3] if row[3] else {}

                result = SearchResult(
                    id=row[0],
                    content=row[1],
                    score=row[6],
                    hierarchy=hierarchy,
                    metadata=metadata,
                    search_type="keyword",
                    keyword_score=row[6],
                    book_title=row[4],
                    book_author=row[5],
                )
                results.append(result)

            logger.info(f"키워드 검색 결과: {len(results)}개")
            return results

        except Exception as e:
            logger.error(f"키워드 검색 중 오류 발생: {e}")
            return []
        finally:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        similarity_threshold: float = 0.3,
    ) -> List[SearchResult]:
        """
        하이브리드 검색을 수행합니다 (의미적 + 키워드).

        Args:
            query: 검색 쿼리
            limit: 반환할 결과 수
            semantic_weight: 의미적 검색 가중치
            keyword_weight: 키워드 검색 가중치
            similarity_threshold: 유사도 임계값

        Returns:
            검색 결과 리스트
        """
        logger.info(f"하이브리드 검색 수행: '{query}'")

        # 쿼리 임베딩 생성
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            logger.warning("쿼리 임베딩을 생성할 수 없어 키워드 검색만 수행합니다.")
            return self.keyword_search(query, limit)

        try:
            conn = psycopg.connect(self.connection_string)
            register_vector(conn)
            cursor = conn.cursor()

            # 하이브리드 검색: 의미적 유사도와 키워드 랭크를 조합
            cursor.execute(
                """
                SELECT 
                    id, content, hierarchy, metadata, book_title, book_author,
                    CASE 
                        WHEN embedding IS NOT NULL THEN
                            (%s * (1 - (embedding <=> %s))) + 
                            (%s * COALESCE(ts_rank(ts_content, plainto_tsquery('simple', %s)), 0))
                        ELSE
                            %s * COALESCE(ts_rank(ts_content, plainto_tsquery('simple', %s)), 0)
                    END as combined_score,
                    CASE 
                        WHEN embedding IS NOT NULL THEN 1 - (embedding <=> %s)
                        ELSE 0
                    END as semantic_score,
                    COALESCE(ts_rank(ts_content, plainto_tsquery('simple', %s)), 0) as keyword_score
                FROM documents
                WHERE 
                    (embedding IS NOT NULL AND 1 - (embedding <=> %s) > %s)
                    OR ts_content @@ plainto_tsquery('simple', %s)
                ORDER BY combined_score DESC
                LIMIT %s
            """,
                (
                    semantic_weight,
                    query_embedding,
                    keyword_weight,
                    query,
                    keyword_weight,
                    query,
                    query_embedding,
                    query,
                    query_embedding,
                    similarity_threshold,
                    query,
                    limit,
                ),
            )

            results = []
            for row in cursor.fetchall():
                # JSONB 타입은 이미 파싱되어 딕셔너리로 반환됨
                hierarchy = row[2] if row[2] else []
                metadata = row[3] if row[3] else {}

                result = SearchResult(
                    id=row[0],
                    content=row[1],
                    score=row[6],  # combined_score
                    hierarchy=hierarchy,
                    metadata=metadata,
                    search_type="hybrid",
                    semantic_score=row[7],  # semantic_score
                    keyword_score=row[8],  # keyword_score
                    book_title=row[4],
                    book_author=row[5],
                )
                results.append(result)

            logger.info(f"하이브리드 검색 결과: {len(results)}개")
            return results

        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류 발생: {e}")
            return []
        finally:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()

    def hierarchical_search(
        self,
        query: str,
        hierarchy_filter: Optional[Dict[str, Any]] = None,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        계층 구조 기반 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            hierarchy_filter: 계층 구조 필터 (예: {'level': 1, 'title': 'Chapter 1'})
            search_type: 검색 유형
            limit: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        logger.info(f"계층 구조 검색 수행: '{query}' (필터: {hierarchy_filter})")

        # 기본 검색 수행
        if search_type == SearchType.SEMANTIC:
            results = self.semantic_search(
                query, limit * 2
            )  # 더 많은 결과를 가져와서 필터링
        elif search_type == SearchType.KEYWORD:
            results = self.keyword_search(query, limit * 2)
        else:
            results = self.hybrid_search(query, limit * 2)

        # 계층 구조 필터 적용
        if hierarchy_filter:
            filtered_results = []
            for result in results:
                if self._match_hierarchy_filter(result.hierarchy, hierarchy_filter):
                    filtered_results.append(result)
            results = filtered_results[:limit]

        logger.info(f"계층 구조 검색 결과: {len(results)}개")
        return results

    def _match_hierarchy_filter(
        self, hierarchy: List[Dict[str, Any]], filter_dict: Dict[str, Any]
    ) -> bool:
        """
        계층 구조가 필터 조건을 만족하는지 확인합니다.

        Args:
            hierarchy: 문서의 계층 구조
            filter_dict: 필터 조건

        Returns:
            필터 조건 만족 여부
        """
        for level_info in hierarchy:
            match = True
            for key, value in filter_dict.items():
                if key not in level_info or level_info[key] != value:
                    match = False
                    break
            if match:
                return True
        return False

    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 10,
        hierarchy_filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        통합 검색 인터페이스입니다.

        Args:
            query: 검색 쿼리
            search_type: 검색 유형
            limit: 반환할 결과 수
            hierarchy_filter: 계층 구조 필터
            **kwargs: 각 검색 방식별 추가 매개변수

        Returns:
            검색 결과 리스트
        """
        if hierarchy_filter:
            return self.hierarchical_search(query, hierarchy_filter, search_type, limit)

        if search_type == SearchType.SEMANTIC:
            return self.semantic_search(query, limit, **kwargs)
        elif search_type == SearchType.KEYWORD:
            return self.keyword_search(query, limit, **kwargs)
        else:  # HYBRID
            return self.hybrid_search(query, limit, **kwargs)

    def get_document_statistics(self) -> Dict[str, Any]:
        """
        데이터베이스의 문서 통계를 반환합니다.

        Returns:
            통계 정보 딕셔너리
        """
        try:
            conn = psycopg.connect(self.connection_string)
            cursor = conn.cursor()

            # 기본 통계
            cursor.execute("""
                SELECT 
                    COUNT(*) as document_count,
                    AVG(LENGTH(content)) as avg_content_length,
                    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as documents_with_embeddings,
                    MIN(LENGTH(content)) as min_content_length,
                    MAX(LENGTH(content)) as max_content_length
                FROM documents
            """)

            stats = cursor.fetchone()

            statistics = {
                "document_count": stats[0] or 0,
                "avg_content_length": float(stats[1]) if stats[1] else 0,
                "documents_with_embeddings": stats[2] or 0,
                "min_content_length": stats[3] or 0,
                "max_content_length": stats[4] or 0,
            }

            # 계층 구조별 통계 (예시)
            cursor.execute("""
                SELECT 
                    hierarchy->>'toc_level' as toc_level,
                    COUNT(*) as count
                FROM documents
                WHERE hierarchy IS NOT NULL
                GROUP BY hierarchy->>'toc_level'
                ORDER BY toc_level
            """)

            level_stats = {}
            for row in cursor.fetchall():
                level = row[0] if row[0] is not None else "unknown"
                level_stats[level] = row[1]

            statistics["by_toc_level"] = level_stats

            return statistics

        except Exception as e:
            logger.error(f"통계 조회 중 오류 발생: {e}")
            return {}
        finally:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()

    def get_database_stats(self) -> Dict[str, Any]:
        """
        데이터베이스 통계를 가져옵니다 (search_system과의 호환성을 위한 별칭).
        """
        return self.get_document_statistics()


def main():
    """메인 함수 - 대화형 검색 인터페이스"""
    import os
    from ..utils.format import format_search_results, interactive_result_viewer

    search_system = SearchSystem(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        connection_string=os.getenv(
            "DATABASE_URL", "postgresql://user@localhost:5432/postgres"
        ),
    )

    print("🔍 대화형 검색 시스템")
    print("검색어를 입력하세요 (종료하려면 'quit' 입력)")

    while True:
        try:
            query = input("\n검색어: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            # 하이브리드 검색 수행
            results = search_system.search(query, SearchType.HYBRID, limit=5)

            if results:
                print(f"\n🎯 검색 결과 ({len(results)}개):")
                print(format_search_results(results))

                # 대화형 탐색 옵션
                explore = input("\n대화형으로 탐색하시겠습니까? (y/N): ").strip()
                if explore.lower() in ["y", "yes"]:
                    interactive_result_viewer(results)
            else:
                print("🔍 검색 결과가 없습니다.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    print("👋 검색 시스템을 종료합니다.")


if __name__ == "__main__":
    main()
