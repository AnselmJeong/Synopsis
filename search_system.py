"""
EPUB 데이터베이스 검색 시스템
의미적 검색, 키워드 검색, 하이브리드 검색을 지원합니다.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import psycopg
from pgvector.psycopg import register_vector
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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


class SearchSystem:
    """EPUB 데이터베이스 검색 시스템"""

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
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.connection_string = connection_string

        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
        else:
            logger.warning(
                "Gemini API 키가 설정되지 않았습니다. 의미적 검색이 제한됩니다."
            )

    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        검색 쿼리에 대한 임베딩을 생성합니다.

        Args:
            query: 검색 쿼리

        Returns:
            쿼리 임베딩 벡터 또는 None
        """
        if not self.gemini_api_key:
            return None

        try:
            result = genai.embed_content(
                model="models/embedding-001", content=query, task_type="retrieval_query"
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

            # 코사인 유사도 기반 검색
            cursor.execute(
                """
                SELECT 
                    id, content, hierarchy, metadata,
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
                result = SearchResult(
                    id=row[0],
                    content=row[1],
                    score=row[4],
                    hierarchy=row[2],
                    metadata=row[3],
                    search_type="semantic",
                    semantic_score=row[4],
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

            # 전문검색 쿼리 구성
            # 공백을 기준으로 단어를 분리하고 OR 조건으로 연결
            search_terms = query.strip().split()
            search_query = " | ".join(search_terms)

            cursor.execute(
                """
                SELECT 
                    id, content, hierarchy, metadata,
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
                result = SearchResult(
                    id=row[0],
                    content=row[1],
                    score=row[4],
                    hierarchy=row[2],
                    metadata=row[3],
                    search_type="keyword",
                    keyword_score=row[4],
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
                    id, content, hierarchy, metadata,
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
                result = SearchResult(
                    id=row[0],
                    content=row[1],
                    score=row[4],  # combined_score
                    hierarchy=row[2],
                    metadata=row[3],
                    search_type="hybrid",
                    semantic_score=row[5],  # semantic_score
                    keyword_score=row[6],  # keyword_score
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
            **kwargs: 추가 검색 옵션

        Returns:
            검색 결과 리스트
        """
        if hierarchy_filter:
            return self.hierarchical_search(query, hierarchy_filter, search_type, limit)

        if search_type == SearchType.SEMANTIC:
            return self.semantic_search(query, limit, **kwargs)
        elif search_type == SearchType.KEYWORD:
            return self.keyword_search(query, limit)
        else:
            return self.hybrid_search(query, limit, **kwargs)

    def get_document_statistics(self) -> Dict[str, Any]:
        """
        데이터베이스의 문서 통계를 반환합니다.

        Returns:
            문서 통계 딕셔너리
        """
        try:
            conn = psycopg.connect(self.connection_string)
            cursor = conn.cursor()

            # 총 문서 수
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]

            # 임베딩이 있는 문서 수
            cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
            docs_with_embeddings = cursor.fetchone()[0]

            # 계층 구조별 문서 수
            cursor.execute("""
                SELECT 
                    jsonb_array_length(hierarchy) as depth,
                    COUNT(*) as count
                FROM documents 
                WHERE hierarchy IS NOT NULL
                GROUP BY depth
                ORDER BY depth
            """)
            depth_stats = dict(cursor.fetchall())

            # 평균 콘텐츠 길이
            cursor.execute("SELECT AVG(LENGTH(content)) FROM documents")
            avg_content_length = cursor.fetchone()[0]

            stats = {
                "total_documents": total_docs,
                "documents_with_embeddings": docs_with_embeddings,
                "embedding_coverage": docs_with_embeddings / total_docs
                if total_docs > 0
                else 0,
                "depth_distribution": depth_stats,
                "average_content_length": round(avg_content_length, 2)
                if avg_content_length
                else 0,
            }

            return stats

        except Exception as e:
            logger.error(f"통계 조회 중 오류 발생: {e}")
            return {}
        finally:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()


def format_hierarchy_path(hierarchy: List[Dict[str, Any]]) -> str:
    """
    계층 구조를 보기 좋은 경로 형태로 포맷팅합니다.

    Args:
        hierarchy: 계층 구조 리스트

    Returns:
        포맷팅된 계층 경로 문자열
    """
    if not hierarchy:
        return "루트"

    path_parts = []
    for level in hierarchy:
        title = level.get("title", "Unknown")
        level_num = level.get("level", 0)
        path_parts.append(f"{title} (Level {level_num})")

    return " > ".join(path_parts)


def format_detailed_hierarchy(hierarchy: List[Dict[str, Any]]) -> str:
    """
    계층 구조를 상세하게 표시합니다.

    Args:
        hierarchy: 계층 구조 리스트

    Returns:
        상세한 계층 구조 문자열
    """
    if not hierarchy:
        return "    📁 루트 레벨"

    output = []
    for i, level in enumerate(hierarchy):
        title = level.get("title", "Unknown")
        level_num = level.get("level", 0)
        indent = "    " + "  " * level_num
        icon = "📁" if i < len(hierarchy) - 1 else "📄"
        output.append(f"{indent}{icon} {title} (Level {level_num})")

    return "\n".join(output)


def format_search_results(
    results: List[SearchResult],
    show_hierarchy: bool = True,
    show_full_content: bool = False,
    max_content_length: int = 200,
    show_detailed_scores: bool = True,
) -> str:
    """
    검색 결과를 보기 좋게 포맷팅합니다.

    Args:
        results: 검색 결과 리스트
        show_hierarchy: 계층 구조 표시 여부
        show_full_content: 전체 콘텐츠 표시 여부
        max_content_length: 최대 콘텐츠 길이 (요약 모드에서)
        show_detailed_scores: 상세 점수 표시 여부

    Returns:
        포맷팅된 검색 결과 문자열
    """
    if not results:
        return "🔍 검색 결과가 없습니다."

    output = []
    output.append(f"🎯 검색 결과 ({len(results)}개):")
    output.append("=" * 80)

    for i, result in enumerate(results, 1):
        output.append(f"\n📑 {i}. 문서 ID: {result.id}")

        # 점수 정보
        score_info = f"📊 점수: {result.score:.4f}"
        if show_detailed_scores and result.search_type == "hybrid":
            if result.semantic_score is not None and result.keyword_score is not None:
                score_info += f" (의미적: {result.semantic_score:.4f}, 키워드: {result.keyword_score:.4f})"
        score_info += f" | 검색 유형: {result.search_type}"
        output.append(score_info)

        # 계층 구조 표시
        if show_hierarchy:
            output.append("🗂️  계층 구조:")
            if result.hierarchy:
                output.append(format_detailed_hierarchy(result.hierarchy))
            else:
                output.append("    📁 루트 레벨")

        # 메타데이터 표시
        if result.metadata:
            output.append("ℹ️  메타데이터:")
            file_path = result.metadata.get("file_path", "N/A")
            toc_level = result.metadata.get("toc_level", "N/A")
            chunk_info = f"{result.metadata.get('chunk_index', 0) + 1}/{result.metadata.get('total_chunks', 1)}"
            output.append(f"    📄 파일: {file_path}")
            output.append(f"    📐 TOC 레벨: {toc_level}")
            output.append(f"    🧩 청크: {chunk_info}")

            if result.metadata.get("anchor"):
                output.append(f"    🔗 앵커: {result.metadata['anchor']}")

        # 콘텐츠 표시
        output.append("📝 콘텐츠:")
        if show_full_content:
            # 전체 콘텐츠 표시
            content_lines = result.content.split("\n")
            for line in content_lines:
                if line.strip():
                    output.append(f"    {line}")
        else:
            # 요약 콘텐츠 표시
            content_preview = result.content[:max_content_length]
            if len(result.content) > max_content_length:
                content_preview += "..."

            # 여러 줄로 나누어 표시
            content_lines = content_preview.split("\n")
            for line in content_lines[:5]:  # 최대 5줄까지만 표시
                if line.strip():
                    output.append(f"    {line}")

            if len(content_lines) > 5:
                output.append("    ...")

            # 전체 콘텐츠 길이 표시
            output.append(f"    📏 전체 길이: {len(result.content):,}자")

        output.append("─" * 80)

    return "\n".join(output)


def interactive_result_viewer(results: List[SearchResult]):
    """
    검색 결과를 대화형으로 탐색할 수 있는 뷰어입니다.

    Args:
        results: 검색 결과 리스트
    """
    if not results:
        print("🔍 검색 결과가 없습니다.")
        return

    print(f"\n🎯 검색 결과 ({len(results)}개):")
    print("=" * 50)

    # 요약 목록 표시
    for i, result in enumerate(results, 1):
        hierarchy_path = format_hierarchy_path(result.hierarchy)
        preview = (
            result.content[:100] + "..."
            if len(result.content) > 100
            else result.content
        )
        print(f"{i}. [점수: {result.score:.3f}] {hierarchy_path}")
        print(f"   {preview}")
        print()

    # 대화형 탐색
    while True:
        try:
            choice = input(
                "상세히 볼 결과 번호를 입력하세요 (1-{}, 'q' 종료): ".format(
                    len(results)
                )
            ).strip()

            if choice.lower() == "q":
                break

            try:
                index = int(choice) - 1
                if 0 <= index < len(results):
                    result = results[index]
                    print("\n" + "=" * 80)
                    print(format_search_results([result], show_full_content=True))
                    print("=" * 80)

                    # 추가 옵션
                    action = (
                        input("\n다른 작업을 선택하세요 (c: 계속, b: 뒤로, q: 종료): ")
                        .strip()
                        .lower()
                    )
                    if action == "q":
                        break
                    elif action == "b":
                        continue
                else:
                    print("잘못된 번호입니다.")
            except ValueError:
                print("숫자를 입력해주세요.")

        except KeyboardInterrupt:
            print("\n검색 결과 탐색을 종료합니다.")
            break


def main():
    """메인 함수 - 대화형 검색 인터페이스"""
    import argparse

    parser = argparse.ArgumentParser(description="EPUB 데이터베이스 검색 시스템")
    parser.add_argument(
        "--db-url",
        default="postgresql://user@localhost:5432/postgres",
        help="PostgreSQL 연결 URL",
    )
    parser.add_argument("--stats", action="store_true", help="데이터베이스 통계 표시")
    parser.add_argument("--query", help="검색 쿼리")
    parser.add_argument(
        "--type",
        choices=["semantic", "keyword", "hybrid"],
        default="hybrid",
        help="검색 유형",
    )
    parser.add_argument("--limit", type=int, default=10, help="반환할 결과 수")
    parser.add_argument("--full-content", action="store_true", help="전체 콘텐츠 표시")
    parser.add_argument(
        "--interactive", action="store_true", help="대화형 결과 뷰어 사용"
    )

    args = parser.parse_args()

    # 검색 시스템 초기화
    search_system = SearchSystem(connection_string=args.db_url)

    # 통계 표시
    if args.stats:
        stats = search_system.get_document_statistics()
        print("📊 데이터베이스 통계:")
        print(f"  📄 총 문서 수: {stats.get('total_documents', 0):,}")
        print(f"  🔗 임베딩 보유 문서: {stats.get('documents_with_embeddings', 0):,}")
        print(f"  📈 임베딩 커버리지: {stats.get('embedding_coverage', 0):.1%}")
        print(f"  📏 평균 콘텐츠 길이: {stats.get('average_content_length', 0):,.1f}자")
        print(f"  🔢 깊이별 분포: {stats.get('depth_distribution', {})}")
        print()

    # 쿼리 검색
    if args.query:
        search_type = SearchType(args.type)
        results = search_system.search(args.query, search_type, args.limit)

        if args.interactive:
            interactive_result_viewer(results)
        else:
            print(format_search_results(results, show_full_content=args.full_content))
    else:
        # 대화형 모드
        print("🔍 EPUB 검색 시스템 - 대화형 모드")
        print("검색 쿼리를 입력하세요 (종료: 'quit' 또는 'exit')")

        while True:
            try:
                query = input("\n🔎 검색 쿼리: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break

                if not query:
                    continue

                # 검색 수행
                search_type = SearchType(args.type)
                results = search_system.search(query, search_type, args.limit)

                # 결과 표시
                if results:
                    if args.interactive:
                        interactive_result_viewer(results)
                    else:
                        print(
                            format_search_results(
                                results, show_full_content=args.full_content
                            )
                        )
                else:
                    print("🔍 검색 결과가 없습니다.")

            except KeyboardInterrupt:
                print("\n검색을 종료합니다.")
                break
            except Exception as e:
                logger.error(f"검색 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
