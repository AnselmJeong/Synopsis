#!/usr/bin/env python3
"""
EPUB 분석 및 검색 시스템 통합 실행 스크립트
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from config import Config, validate_config
from database_setup import setup_database
from epub_processor import EPUBProcessor
from search_system import (
    SearchSystem,
    SearchType,
    format_search_results,
    interactive_result_viewer,
)

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_command(args):
    """데이터베이스 설정 명령"""
    try:
        config = Config()
        validate_config(config)
        setup_database(config.database_url)
        print("✅ 데이터베이스 설정이 완료되었습니다.")
    except Exception as e:
        logger.error(f"데이터베이스 설정 중 오류 발생: {e}")
        return 1
    return 0


def process_command(args):
    """EPUB 파일 처리 명령"""
    try:
        if not os.path.exists(args.epub_file):
            print(f"❌ EPUB 파일을 찾을 수 없습니다: {args.epub_file}")
            return 1

        config = Config()
        validate_config(config)

        processor = EPUBProcessor(
            gemini_api_key=config.gemini_api_key,
            connection_string=config.database_url,
            target_chunk_size=args.target_chunk_size,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
        )

        print(f"📖 EPUB 파일 처리 시작: {args.epub_file}")
        print(f"   - 목표 청크 크기: {args.target_chunk_size}자")
        print(f"   - 최소 청크 크기: {args.min_chunk_size}자")
        print(f"   - 최대 청크 크기: {args.max_chunk_size}자")

        processor.process_epub(
            epub_file_path=args.epub_file, save_toc_json=not args.no_toc_json
        )

        print("✅ EPUB 파일 처리가 완료되었습니다.")

    except Exception as e:
        logger.error(f"EPUB 파일 처리 중 오류 발생: {e}")
        return 1
    return 0


def search_command(args):
    """검색 명령"""
    try:
        config = Config()
        validate_config(config)

        search_system = SearchSystem(
            connection_string=config.database_url, gemini_api_key=config.gemini_api_key
        )

        # 검색 타입 결정
        if args.search_type == "semantic":
            search_type = SearchType.SEMANTIC
        elif args.search_type == "keyword":
            search_type = SearchType.KEYWORD
        elif args.search_type == "hybrid":
            search_type = SearchType.HYBRID
        else:
            search_type = SearchType.HYBRID  # 기본값

        # 검색 실행
        results = search_system.search(
            query=args.query,
            search_type=search_type,
            limit=args.limit,
            hierarchy_filter=args.hierarchy_filter,
        )

        if not results:
            print("🔍 검색 결과가 없습니다.")
            return 0

        print(f"🔍 검색 결과 ({len(results)}개):")
        print("=" * 80)

        # 대화형 모드
        if args.interactive:
            interactive_result_viewer(results)
        else:
            # 결과 포맷팅 및 출력
            formatted_results = format_search_results(
                results,
                show_full_content=args.full_content,
                show_detailed_scores=args.detailed_scores,
            )
            print(formatted_results)

    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        return 1
    return 0


def interactive_search():
    """대화형 검색 모드"""
    try:
        config = Config()
        validate_config(config)

        search_system = SearchSystem(
            connection_string=config.database_url, gemini_api_key=config.gemini_api_key
        )

        print("🔍 대화형 검색 모드")
        print("특수 명령어:")
        print("  !stats - 데이터베이스 통계")
        print("  !help - 도움말")
        print("  !interactive - 대화형 뷰어 토글")
        print("  exit, quit - 종료")
        print("=" * 50)

        interactive_mode = False

        while True:
            try:
                query = input("\n검색어를 입력하세요: ").strip()

                if not query or query.lower() in ["exit", "quit"]:
                    print("👋 검색을 종료합니다.")
                    break

                # 특수 명령어 처리
                if query.startswith("!"):
                    if query == "!stats":
                        stats = search_system.get_database_stats()
                        print(f"📊 데이터베이스 통계:")
                        print(f"   - 총 문서 수: {stats.get('document_count', 0):,}")
                        print(
                            f"   - 평균 문서 길이: {stats.get('avg_content_length', 0):.0f}자"
                        )
                        print(
                            f"   - 임베딩 생성된 문서: {stats.get('documents_with_embeddings', 0):,}"
                        )
                        continue
                    elif query == "!help":
                        print("🔍 검색 도움말:")
                        print("   - 일반 검색: 검색어 입력")
                        print("   - 검색 타입은 자동으로 하이브리드(의미적+키워드)")
                        print("   - 대화형 모드에서는 결과를 상세히 탐색할 수 있습니다")
                        continue
                    elif query == "!interactive":
                        interactive_mode = not interactive_mode
                        print(
                            f"🎮 대화형 뷰어: {'켜짐' if interactive_mode else '꺼짐'}"
                        )
                        continue
                    else:
                        print("❓ 알 수 없는 명령어입니다. !help를 참조하세요.")
                        continue

                # 검색 실행
                print(f"🔍 검색 중: '{query}'...")
                results = search_system.search(
                    query=query, search_type=SearchType.HYBRID, limit=10
                )

                if not results:
                    print("   검색 결과가 없습니다.")
                    continue

                print(f"   {len(results)}개의 결과를 찾았습니다.")

                if interactive_mode:
                    interactive_result_viewer(results)
                else:
                    formatted_results = format_search_results(
                        results, show_full_content=False, show_detailed_scores=True
                    )
                    print(formatted_results)

            except KeyboardInterrupt:
                print("\n👋 검색을 종료합니다.")
                break
            except Exception as e:
                logger.error(f"검색 중 오류 발생: {e}")
                continue

    except Exception as e:
        logger.error(f"대화형 검색 모드 실행 중 오류 발생: {e}")
        return 1
    return 0


def create_parser():
    """명령행 인수 파서 생성"""
    parser = argparse.ArgumentParser(
        description="EPUB 분석 및 검색 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 데이터베이스 설정
  python main.py setup
  
  # EPUB 파일 처리 (계층적 청킹)
  python main.py process book.epub --target-chunk-size 4096
  
  # 하이브리드 검색
  python main.py search "인공지능" --search-type hybrid
  
  # 대화형 검색
  python main.py interactive
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령어")

    # setup 명령
    subparsers.add_parser("setup", help="데이터베이스 설정")

    # process 명령
    process_parser = subparsers.add_parser("process", help="EPUB 파일 처리")
    process_parser.add_argument("epub_file", help="처리할 EPUB 파일 경로")
    process_parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=4096,
        help="목표 청크 크기 (기본값: 4096)",
    )
    process_parser.add_argument(
        "--min-chunk-size", type=int, default=2048, help="최소 청크 크기 (기본값: 2048)"
    )
    process_parser.add_argument(
        "--max-chunk-size", type=int, default=8192, help="최대 청크 크기 (기본값: 8192)"
    )
    process_parser.add_argument(
        "--no-toc-json", action="store_true", help="TOC JSON 파일 저장 건너뛰기"
    )

    # search 명령
    search_parser = subparsers.add_parser("search", help="콘텐츠 검색")
    search_parser.add_argument("query", help="검색 질의")
    search_parser.add_argument(
        "--search-type",
        choices=["semantic", "keyword", "hybrid"],
        default="hybrid",
        help="검색 타입 (기본값: hybrid)",
    )
    search_parser.add_argument(
        "--limit", type=int, default=10, help="검색 결과 개수 (기본값: 10)"
    )
    search_parser.add_argument(
        "--hierarchy-filter", type=str, help="계층 구조 필터 (JSON 형식)"
    )
    search_parser.add_argument(
        "--full-content", action="store_true", help="전체 콘텐츠 표시"
    )
    search_parser.add_argument(
        "--detailed-scores", action="store_true", help="상세 점수 정보 표시"
    )
    search_parser.add_argument(
        "--interactive", action="store_true", help="대화형 결과 뷰어 사용"
    )

    # interactive 명령
    subparsers.add_parser("interactive", help="대화형 검색 모드")

    return parser


def main():
    """메인 함수"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # 명령 실행
    if args.command == "setup":
        return setup_command(args)
    elif args.command == "process":
        return process_command(args)
    elif args.command == "search":
        return search_command(args)
    elif args.command == "interactive":
        return interactive_search()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
