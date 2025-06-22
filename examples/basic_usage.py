#!/usr/bin/env python3
"""
Synopsis 패키지 기본 사용 예제

이 예제는 Synopsis를 Python 라이브러리로 사용하는 방법을 보여줍니다.
"""

import os
from pathlib import Path

# Synopsis 패키지 import
from synopsis import (
    EPUBProcessor,
    SearchSystem,
    SearchType,
    setup_database,
    Config,
    validate_config,
    format_search_results,
)


def main():
    """기본 사용 예제"""
    print("📚 Synopsis 패키지 기본 사용 예제")
    print("=" * 50)

    # 1. 설정 로드
    try:
        config = Config()
        validate_config(config)
        print("✅ 설정 로드 완료")
    except Exception as e:
        print(f"❌ 설정 오류: {e}")
        return

    # 2. 데이터베이스 설정 (필요한 경우)
    try:
        setup_database(config.database_url)
        print("✅ 데이터베이스 설정 완료")
    except Exception as e:
        print(f"❌ 데이터베이스 설정 오류: {e}")
        return

    # 3. EPUB 파일 처리 (파일이 있는 경우)
    epub_file = "book/The Elegant Universe..epub"
    if Path(epub_file).exists():
        print(f"\n📖 EPUB 파일 처리: {epub_file}")

        processor = EPUBProcessor(
            gemini_api_key=config.gemini_api_key,
            connection_string=config.database_url,
            target_chunk_size=4096,
            max_chunk_size=8192,
        )

        try:
            processor.process_epub(epub_file, save_toc_json=True)
            print("✅ EPUB 파일 처리 완료")
        except Exception as e:
            print(f"❌ EPUB 처리 오류: {e}")
    else:
        print(f"\n⚠️  EPUB 파일을 찾을 수 없습니다: {epub_file}")

    # 4. 검색 시스템 사용
    print("\n🔍 검색 시스템 테스트")

    search_system = SearchSystem(
        gemini_api_key=config.gemini_api_key,
        connection_string=config.database_url,
    )

    # 데이터베이스 통계 확인
    try:
        stats = search_system.get_database_stats()
        print(f"📊 데이터베이스 통계:")
        print(f"   - 총 문서 수: {stats.get('document_count', 0):,}")
        print(f"   - 평균 문서 길이: {stats.get('avg_content_length', 0):.0f}자")
        print(f"   - 임베딩 보유 문서: {stats.get('documents_with_embeddings', 0):,}")
    except Exception as e:
        print(f"❌ 통계 조회 오류: {e}")

    # 5. 다양한 검색 방식 테스트
    test_queries = [
        "우주의 구조",
        "양자역학",
        "상대성이론",
    ]

    for query in test_queries:
        print(f"\n🔎 검색어: '{query}'")

        try:
            # 하이브리드 검색
            results = search_system.search(
                query=query, search_type=SearchType.HYBRID, limit=3
            )

            if results:
                print(f"   ✅ {len(results)}개 결과 발견")

                # 간단한 결과 출력
                for i, result in enumerate(results[:2], 1):
                    content_preview = (
                        result.content[:100] + "..."
                        if len(result.content) > 100
                        else result.content
                    )
                    print(f"   {i}. [점수: {result.score:.3f}] {content_preview}")
            else:
                print("   🔍 결과 없음")

        except Exception as e:
            print(f"   ❌ 검색 오류: {e}")

    print("\n✨ 예제 완료!")


def advanced_usage():
    """고급 사용 예제"""
    print("\n📚 Synopsis 패키지 고급 사용 예제")
    print("=" * 50)

    config = Config()
    search_system = SearchSystem(
        gemini_api_key=config.gemini_api_key,
        connection_string=config.database_url,
    )

    # 1. 의미적 검색만 사용
    print("\n🧠 의미적 검색:")
    results = search_system.semantic_search("물리학의 기본 원리", limit=5)
    if results:
        print(f"   {len(results)}개 결과")
        for result in results[:2]:
            print(f"   - 점수: {result.semantic_score:.3f}")

    # 2. 키워드 검색만 사용
    print("\n🔤 키워드 검색:")
    results = search_system.keyword_search("우주", limit=5)
    if results:
        print(f"   {len(results)}개 결과")
        for result in results[:2]:
            print(f"   - 점수: {result.keyword_score:.3f}")

    # 3. 계층 구조 필터링
    print("\n🌳 계층 구조 검색:")
    hierarchy_filter = {"toc_level": "0"}  # 최상위 레벨만
    results = search_system.hierarchical_search(
        "물질", hierarchy_filter=hierarchy_filter, limit=3
    )
    if results:
        print(f"   {len(results)}개 결과 (최상위 레벨)")

    # 4. 사용자 정의 하이브리드 가중치
    print("\n⚖️  가중치 조정 하이브리드 검색:")
    results = search_system.hybrid_search(
        "에너지", semantic_weight=0.8, keyword_weight=0.2, limit=3
    )
    if results:
        print(f"   {len(results)}개 결과 (의미적 80%, 키워드 20%)")


if __name__ == "__main__":
    main()

    # 고급 예제도 실행하려면 주석 해제
    # advanced_usage()
