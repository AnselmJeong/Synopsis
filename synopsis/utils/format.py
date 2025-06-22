"""
검색 결과 포맷팅 및 대화형 뷰어 유틸리티
"""

import json
from typing import List, Dict, Any

from ..core.search import SearchResult


def format_hierarchy_path(hierarchy: List[Dict[str, Any]]) -> str:
    """
    계층 구조를 경로 형태로 포맷팅합니다.

    Args:
        hierarchy: 계층 구조 정보

    Returns:
        포맷팅된 계층 경로
    """
    if not hierarchy:
        return "알 수 없음"

    try:
        # 계층 구조에서 title들을 추출
        hierarchy_data = (
            hierarchy.get("toc_hierarchy", [])
            if isinstance(hierarchy, dict)
            else hierarchy
        )

        if not hierarchy_data:
            return "알 수 없음"

        path_parts = []
        for level_info in hierarchy_data:
            if "title" in level_info:
                title = level_info["title"]
                level = level_info.get("level", 0)
                path_parts.append(f"{'  ' * level}{title}")

        return " > ".join(path_parts) if path_parts else "알 수 없음"

    except Exception:
        return "알 수 없음"


def format_detailed_hierarchy(hierarchy: List[Dict[str, Any]]) -> str:
    """
    계층 구조를 상세하게 포맷팅합니다.

    Args:
        hierarchy: 계층 구조 정보

    Returns:
        상세 포맷팅된 계층 구조
    """
    if not hierarchy:
        return "계층 정보 없음"

    try:
        if isinstance(hierarchy, dict):
            result = []
            for key, value in hierarchy.items():
                if key == "toc_hierarchy" and isinstance(value, list):
                    result.append("📖 TOC 계층:")
                    for level_info in value:
                        level = level_info.get("level", 0)
                        title = level_info.get("title", "제목 없음")
                        indent = "  " * level
                        result.append(f"{indent}├─ 레벨 {level}: {title}")
                else:
                    result.append(f"{key}: {value}")
            return "\n".join(result)
        else:
            return json.dumps(hierarchy, indent=2, ensure_ascii=False)
    except Exception:
        return "계층 정보 파싱 오류"


def format_search_results(
    results: List[SearchResult],
    show_hierarchy: bool = True,
    show_full_content: bool = False,
    max_content_length: int = 200,
    show_detailed_scores: bool = True,
) -> str:
    """
    검색 결과를 포맷팅합니다.

    Args:
        results: 검색 결과 리스트
        show_hierarchy: 계층 구조 표시 여부
        show_full_content: 전체 콘텐츠 표시 여부
        max_content_length: 최대 콘텐츠 길이
        show_detailed_scores: 상세 점수 정보 표시 여부

    Returns:
        포맷팅된 검색 결과 문자열
    """
    if not results:
        return "검색 결과가 없습니다."

    formatted_results = []

    for i, result in enumerate(results, 1):
        result_lines = []

        # 헤더
        result_lines.append(f"📄 결과 {i} (ID: {result.id})")

        # 책 정보 추가
        if result.book_title:
            book_info = f"📚 {result.book_title}"
            if result.book_author:
                book_info += f" - {result.book_author}"
            result_lines.append(book_info)

        result_lines.append("=" * 50)

        # 점수 정보
        if show_detailed_scores:
            score_info = f"🎯 전체 점수: {result.score:.4f}"
            if result.search_type == "hybrid":
                if result.semantic_score is not None:
                    score_info += f" | 의미적: {result.semantic_score:.4f}"
                if result.keyword_score is not None:
                    score_info += f" | 키워드: {result.keyword_score:.4f}"
            result_lines.append(score_info)

        # 계층 구조
        if show_hierarchy:
            hierarchy_path = format_hierarchy_path(result.hierarchy)
            result_lines.append(f"📍 위치: {hierarchy_path}")

        # 콘텐츠
        content = result.content
        if not show_full_content and len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        result_lines.append(f"📝 내용:")
        result_lines.append(content)

        # 메타데이터 (간략하게)
        if result.metadata:
            chunk_info = []
            if "chunk_index" in result.metadata:
                chunk_info.append(f"청크 {result.metadata['chunk_index'] + 1}")
            if "total_chunks" in result.metadata:
                chunk_info.append(f"총 {result.metadata['total_chunks']}개")
            if "content_length" in result.metadata:
                chunk_info.append(f"{result.metadata['content_length']}자")

            if chunk_info:
                result_lines.append(f"ℹ️  메타데이터: {' | '.join(chunk_info)}")

        result_lines.append("")  # 빈 줄
        formatted_results.append("\n".join(result_lines))

    return "\n".join(formatted_results)


def interactive_result_viewer(results: List[SearchResult]):
    """
    검색 결과를 대화형으로 탐색할 수 있는 뷰어입니다.

    Args:
        results: 검색 결과 리스트
    """
    if not results:
        print("표시할 검색 결과가 없습니다.")
        return

    current_index = 0

    def show_current_result():
        """현재 결과를 표시합니다."""
        result = results[current_index]
        print("\n" + "=" * 80)
        print(f"📄 결과 {current_index + 1} / {len(results)} (ID: {result.id})")

        # 책 정보 표시
        if result.book_title:
            book_info = f"📚 {result.book_title}"
            if result.book_author:
                book_info += f" - {result.book_author}"
            print(book_info)

        print("=" * 80)

        # 점수 정보
        print(f"🎯 점수: {result.score:.4f} ({result.search_type})")
        if result.search_type == "hybrid":
            if result.semantic_score is not None:
                print(f"   의미적 점수: {result.semantic_score:.4f}")
            if result.keyword_score is not None:
                print(f"   키워드 점수: {result.keyword_score:.4f}")

        # 계층 구조
        hierarchy_path = format_hierarchy_path(result.hierarchy)
        print(f"📍 위치: {hierarchy_path}")

        # 콘텐츠 미리보기
        content_preview = (
            result.content[:300] + "..."
            if len(result.content) > 300
            else result.content
        )
        print(f"\n📝 내용 미리보기:")
        print(content_preview)

        # 메타데이터
        if result.metadata:
            chunk_info = []
            if "chunk_index" in result.metadata:
                chunk_info.append(f"청크 {result.metadata['chunk_index'] + 1}")
            if "total_chunks" in result.metadata:
                chunk_info.append(f"총 {result.metadata['total_chunks']}개")
            if "content_length" in result.metadata:
                chunk_info.append(f"{result.metadata['content_length']}자")

            if chunk_info:
                print(f"\nℹ️  메타데이터: {' | '.join(chunk_info)}")

    def show_help():
        """도움말을 표시합니다."""
        print("\n📖 대화형 뷰어 명령어:")
        print("  n, next     - 다음 결과")
        print("  p, prev     - 이전 결과")
        print("  f, full     - 전체 내용 보기")
        print("  h, hier     - 상세 계층 구조 보기")
        print("  m, meta     - 상세 메타데이터 보기")
        print("  s, summary  - 결과 요약 보기")
        print("  g, goto     - 특정 결과로 이동")
        print("  help        - 이 도움말 보기")
        print("  q, quit     - 종료")

    print(f"🔍 검색 결과 대화형 뷰어 ({len(results)}개 결과)")
    show_help()
    show_current_result()

    while True:
        try:
            command = (
                input("\n명령어를 입력하세요 (help를 입력하면 도움말): ")
                .strip()
                .lower()
            )

            if command in ["q", "quit", "exit"]:
                print("👋 뷰어를 종료합니다.")
                break

            elif command in ["n", "next"]:
                if current_index < len(results) - 1:
                    current_index += 1
                    show_current_result()
                else:
                    print("❗ 마지막 결과입니다.")

            elif command in ["p", "prev"]:
                if current_index > 0:
                    current_index -= 1
                    show_current_result()
                else:
                    print("❗ 첫 번째 결과입니다.")

            elif command in ["f", "full"]:
                result = results[current_index]
                print(f"\n📄 전체 내용 (결과 {current_index + 1}):")
                print("-" * 80)
                print(result.content)
                print("-" * 80)

            elif command in ["h", "hier"]:
                result = results[current_index]
                print(f"\n🌳 상세 계층 구조 (결과 {current_index + 1}):")
                print("-" * 80)
                print(format_detailed_hierarchy(result.hierarchy))
                print("-" * 80)

            elif command in ["m", "meta"]:
                result = results[current_index]
                print(f"\n📊 상세 메타데이터 (결과 {current_index + 1}):")
                print("-" * 80)
                print(json.dumps(result.metadata, indent=2, ensure_ascii=False))
                print("-" * 80)

            elif command in ["s", "summary"]:
                print(f"\n📈 검색 결과 요약:")
                print(f"   총 결과 수: {len(results)}")
                if results:
                    scores = [r.score for r in results]
                    print(f"   평균 점수: {sum(scores) / len(scores):.4f}")
                    print(f"   최고 점수: {max(scores):.4f}")
                    print(f"   최저 점수: {min(scores):.4f}")

                    search_types = {}
                    for r in results:
                        search_types[r.search_type] = (
                            search_types.get(r.search_type, 0) + 1
                        )
                    print(f"   검색 유형별: {search_types}")

            elif command in ["g", "goto"]:
                try:
                    target = input(f"이동할 결과 번호 (1-{len(results)}): ").strip()
                    target_index = int(target) - 1
                    if 0 <= target_index < len(results):
                        current_index = target_index
                        show_current_result()
                    else:
                        print(f"❗ 올바른 범위의 번호를 입력하세요 (1-{len(results)}).")
                except ValueError:
                    print("❗ 올바른 숫자를 입력하세요.")

            elif command == "help":
                show_help()

            else:
                print(
                    "❓ 알 수 없는 명령어입니다. 'help'를 입력하면 도움말을 볼 수 있습니다."
                )

        except KeyboardInterrupt:
            print("\n👋 뷰어를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")
