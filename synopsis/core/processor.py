"""
EPUB 파일 분석 및 PostgreSQL 저장 시스템
계층적 TOC 구조를 추출하고 Gemini 임베딩을 생성하여 데이터베이스에 저장합니다.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import psycopg
from pgvector.psycopg import register_vector
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

from ..models.toc import TOCEntry
from ..models.chunk import ContentChunk
from .database import setup_database

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)


class EPUBProcessor:
    """EPUB 파일을 처리하는 메인 클래스"""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        connection_string: str = "postgresql://user@localhost:5432/postgres",
        target_chunk_size: int = 4096,
        min_chunk_size: int = 1000,
        max_chunk_size: int = 6000,
    ):
        """
        EPUB 프로세서를 초기화합니다.

        Args:
            gemini_api_key: Gemini API 키
            connection_string: PostgreSQL 연결 문자열
            target_chunk_size: 목표 청크 크기 (문자 단위)
            min_chunk_size: 최소 청크 크기
            max_chunk_size: 최대 청크 크기
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.connection_string = connection_string
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        logger.info(
            f"EPUB 프로세서 초기화: 목표={target_chunk_size}, 최소={min_chunk_size}, 최대={max_chunk_size}"
        )

        if not self.gemini_api_key:
            logger.warning(
                "Gemini API 키가 설정되지 않았습니다. 임베딩 생성이 건너뛰어집니다."
            )
        else:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-pro")

    def extract_toc_hierarchy(self, epub_file_path: str) -> List[TOCEntry]:
        """
        EPUB 파일에서 계층적 TOC를 추출합니다.

        Args:
            epub_file_path: EPUB 파일 경로

        Returns:
            TOC 항목들의 리스트
        """
        logger.info(f"EPUB 파일에서 TOC를 추출하는 중: {epub_file_path}")

        try:
            book = epub.read_epub(epub_file_path)
            toc_entries = []
            current_hierarchy = []

            def process_toc_item(item, level: int = 0):
                """TOC 항목을 재귀적으로 처리합니다."""
                nonlocal current_hierarchy, toc_entries

                if isinstance(item, tuple):
                    # (Section, [NavPoint, ...]) 형태
                    section, nav_points = item
                    if hasattr(section, "title"):
                        title = section.title
                    else:
                        title = str(section)

                    # 현재 계층에 추가
                    hierarchy_entry = {"level": level, "title": title}
                    current_hierarchy = current_hierarchy[:level] + [hierarchy_entry]

                    # NavPoint 처리
                    for nav_point in nav_points:
                        process_toc_item(nav_point, level + 1)

                elif hasattr(item, "title") and hasattr(item, "href"):
                    # NavPoint 객체
                    title = item.title.strip()
                    href = item.href

                    # 파일 경로와 앵커 분리
                    if "#" in href:
                        file_path, anchor = href.split("#", 1)
                    else:
                        file_path, anchor = href, None

                    # 현재 계층에 추가
                    hierarchy_entry = {"level": level, "title": title}
                    current_hierarchy = current_hierarchy[:level] + [hierarchy_entry]

                    # TOC 엔트리 생성
                    toc_entry = TOCEntry(
                        title=title,
                        level=level,
                        file_path=file_path,
                        hierarchy=list(current_hierarchy),
                        anchor=anchor,
                    )
                    toc_entries.append(toc_entry)

                    logger.debug(f"TOC 항목 추가: {title} (레벨 {level})")

            # TOC 처리
            for toc_item in book.toc:
                process_toc_item(toc_item)

            logger.info(f"총 {len(toc_entries)}개의 TOC 항목을 추출했습니다.")
            return toc_entries

        except Exception as e:
            logger.error(f"TOC 추출 중 오류 발생: {e}")
            raise

    def extract_content_from_file(self, book: epub.EpubBook, file_path: str) -> str:
        """
        EPUB에서 특정 파일의 텍스트 콘텐츠를 추출합니다. (간단한 버전)

        Args:
            book: EPUB 책 객체
            file_path: 파일 경로

        Returns:
            추출된 텍스트 콘텐츠
        """
        logger.debug(f"파일에서 콘텐츠 추출 중: {file_path}")

        try:
            # 파일 경로 정규화
            if file_path.startswith("/"):
                file_path = file_path[1:]

            # 아이템 찾기
            item = None
            for book_item in book.get_items():
                if book_item.get_type() == ebooklib.ITEM_DOCUMENT:
                    if (
                        book_item.get_name() == file_path
                        or book_item.file_name == file_path
                        or book_item.get_name().endswith(file_path)
                    ):
                        item = book_item
                        break

            if not item:
                logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
                return ""

            # HTML 콘텐츠 파싱
            html_content = item.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, "html.parser")

            # 스크립트와 스타일 태그 제거
            for script in soup(["script", "style"]):
                script.decompose()

            # 텍스트만 추출
            text = soup.get_text()

            # 텍스트 정리
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            logger.debug(f"추출된 콘텐츠 길이: {len(text)}자")
            return text

        except Exception as e:
            logger.error(f"콘텐츠 추출 중 오류 발생: {e}")
            return ""

    def create_smart_chunks(
        self, content: str, toc_entry: TOCEntry
    ) -> List[ContentChunk]:
        """
        스마트 청킹: 자연스러운 소제목을 감지하여 청크를 생성합니다.

        Args:
            content: 청킹할 콘텐츠
            toc_entry: TOC 항목

        Returns:
            생성된 청크들의 리스트
        """
        logger.info(f"스마트 청킹 시작: {toc_entry.title}")

        if not content.strip():
            logger.warning("빈 콘텐츠입니다.")
            return []

        try:
            # 1. 자연스러운 소제목 감지
            sections = self._detect_natural_subtitles(content)
            logger.info(f"{len(sections)}개의 섹션을 감지했습니다.")

            # 2. 각 섹션을 청크로 변환
            all_chunks = []
            for i, section in enumerate(sections):
                section_chunks = self._create_chunks_from_section(
                    section, toc_entry, len(all_chunks)
                )
                all_chunks.extend(section_chunks)

            # 3. 크기 검증 및 강제 분할
            validated_chunks = []
            for chunk in all_chunks:
                if len(chunk.content) <= self.max_chunk_size:
                    validated_chunks.append(chunk)
                else:
                    logger.warning(
                        f"청크 크기 초과 ({len(chunk.content)}자), 강제 분할 실행"
                    )
                    split_contents = self._force_split_content(chunk.content)
                    for j, split_content in enumerate(split_contents):
                        new_chunk = ContentChunk(
                            content=split_content,
                            toc_entry=chunk.toc_entry,
                            chunk_index=len(validated_chunks),
                            total_chunks=0,  # 나중에 업데이트
                            section_info={
                                **chunk.section_info,
                                "force_split": True,
                                "force_split_part": j + 1,
                                "force_split_total": len(split_contents),
                            },
                        )
                        validated_chunks.append(new_chunk)

            # 4. total_chunks 업데이트
            total_chunks = len(validated_chunks)
            for chunk in validated_chunks:
                chunk.total_chunks = total_chunks

            logger.info(f"스마트 청킹 완료: {total_chunks}개 청크 생성")
            return validated_chunks

        except Exception as e:
            logger.error(f"스마트 청킹 중 오류 발생: {e}")
            # 폴백: 간단한 청킹
            logger.info("간단한 청킹으로 폴백")
            return self._create_simple_chunks_fallback(content, toc_entry)

    def _detect_natural_subtitles(self, content: str) -> List[Dict[str, Any]]:
        """
        자연스러운 소제목을 감지하여 섹션으로 나눕니다.

        Args:
            content: 분석할 콘텐츠

        Returns:
            섹션 정보의 리스트
        """
        logger.debug("자연스러운 소제목 감지 중...")

        paragraphs = content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [{"title": "전체 콘텐츠", "content": content, "start_idx": 0}]

        sections = []
        current_section_start = 0
        current_section_title = "서론"

        for i, paragraph in enumerate(paragraphs):
            if self._is_natural_subtitle(paragraph, i, paragraphs):
                # 이전 섹션 마무리
                if i > current_section_start:
                    section_content = "\n\n".join(paragraphs[current_section_start:i])
                    sections.append(
                        {
                            "title": current_section_title,
                            "content": section_content,
                            "start_idx": current_section_start,
                            "end_idx": i - 1,
                        }
                    )

                # 새 섹션 시작
                current_section_title = paragraph
                current_section_start = i + 1

        # 마지막 섹션 처리
        if current_section_start < len(paragraphs):
            section_content = "\n\n".join(paragraphs[current_section_start:])
            sections.append(
                {
                    "title": current_section_title,
                    "content": section_content,
                    "start_idx": current_section_start,
                    "end_idx": len(paragraphs) - 1,
                }
            )

        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections:
            sections = [{"title": "전체 콘텐츠", "content": content, "start_idx": 0}]

        logger.debug(f"{len(sections)}개의 섹션 감지 완료")
        return sections

    def _is_natural_subtitle(
        self, paragraph: str, index: int, all_paragraphs: List[str]
    ) -> bool:
        """
        문단이 자연스러운 소제목인지 판단합니다.

        Args:
            paragraph: 검사할 문단
            index: 문단의 인덱스
            all_paragraphs: 전체 문단 목록

        Returns:
            소제목 여부
        """
        # 1. 기본 길이 체크
        if len(paragraph) > 80:
            return False

        # 2. 숫자로 시작하는 패턴 (예: "1. 서론", "2.3 양자역학")
        if re.match(r"^\d+[\.\)]\s+", paragraph) or re.match(
            r"^\d+\.\d+[\.\)]\s+", paragraph
        ):
            return True

        # 3. 특정 키워드로 시작
        subtitle_keywords = [
            "가장",
            "새로운",
            "첫 번째",
            "두 번째",
            "세 번째",
            "마지막",
            "결론",
            "서론",
            "개요",
            "요약",
            "The",
            "New",
            "First",
            "Second",
            "Third",
            "Last",
            "Conclusion",
            "Introduction",
            "Overview",
            "Summary",
        ]

        for keyword in subtitle_keywords:
            if paragraph.startswith(keyword):
                return True

        # 4. 길이와 구조적 특징
        if len(paragraph) < 80:
            # 콜론으로 끝나는 경우
            if paragraph.endswith(":"):
                return True

            # 다음 문단이 현재보다 상당히 긴 경우
            if index + 1 < len(all_paragraphs):
                next_paragraph = all_paragraphs[index + 1]
                if len(next_paragraph) > len(paragraph) * 3:
                    return True

        return False

    def _create_chunks_from_section(
        self, section: Dict[str, Any], toc_entry: TOCEntry, start_index: int
    ) -> List[ContentChunk]:
        """
        섹션에서 청크를 생성합니다.

        Args:
            section: 섹션 정보
            toc_entry: TOC 항목
            start_index: 시작 청크 인덱스

        Returns:
            생성된 청크들의 리스트
        """
        content = section["content"]
        section_title = section["title"]

        if not content.strip():
            return []

        chunks = []

        if len(content) <= self.target_chunk_size:
            # 단일 청크로 충분
            chunk = ContentChunk(
                content=content,
                toc_entry=toc_entry,
                chunk_index=start_index,
                total_chunks=0,  # 나중에 업데이트
                section_info={
                    "section_title": section_title,
                    "section_index": section.get("start_idx", 0),
                    "is_single_chunk": True,
                },
            )
            chunks.append(chunk)
        else:
            # 여러 청크로 분할
            content_parts = self._split_content_preserving_paragraphs(content)

            for i, part in enumerate(content_parts):
                chunk = ContentChunk(
                    content=part,
                    toc_entry=toc_entry,
                    chunk_index=start_index + i,
                    total_chunks=0,  # 나중에 업데이트
                    section_info={
                        "section_title": section_title,
                        "section_index": section.get("start_idx", 0),
                        "chunk_part": i + 1,
                        "chunk_total_parts": len(content_parts),
                    },
                )
                chunks.append(chunk)

        logger.debug(f"섹션 '{section_title}'에서 {len(chunks)}개 청크 생성")
        return chunks

    def _split_content_preserving_paragraphs(self, content: str) -> List[str]:
        """
        문단 경계를 보존하면서 콘텐츠를 분할합니다.

        Args:
            content: 분할할 콘텐츠

        Returns:
            분할된 콘텐츠 부분들
        """
        paragraphs = content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        parts = []
        current_part = ""

        for paragraph in paragraphs:
            # 문단을 추가했을 때 크기 체크
            test_content = (
                current_part + "\n\n" + paragraph if current_part else paragraph
            )

            if len(test_content) <= self.target_chunk_size:
                current_part = test_content
            else:
                # 현재 파트 저장
                if current_part:
                    parts.append(current_part)

                # 새 파트 시작
                current_part = paragraph

        # 마지막 파트 저장
        if current_part:
            parts.append(current_part)

        return parts

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        문장 단위로 텍스트를 분할합니다.

        Args:
            text: 분할할 텍스트

        Returns:
            분할된 문장들
        """
        # 한국어와 영어 문장 분리 패턴
        sentence_endings = r"[.!?。！？]\s+"
        sentences = re.split(sentence_endings, text.strip())

        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_simple_chunks_fallback(
        self, content: str, toc_entry: TOCEntry
    ) -> List[ContentChunk]:
        """
        스마트 청킹이 실패했을 때 사용하는 간단한 폴백 청킹

        Args:
            content: 청킹할 콘텐츠
            toc_entry: TOC 항목

        Returns:
            생성된 청크들의 리스트
        """
        logger.info("간단한 폴백 청킹 실행")

        if not content.strip():
            return []

        chunks = []
        remaining_content = content

        chunk_index = 0
        while remaining_content:
            # 목표 크기만큼 자르기
            if len(remaining_content) <= self.target_chunk_size:
                chunk_content = remaining_content
                remaining_content = ""
            else:
                # 문단 경계에서 자르기 시도
                paragraphs = remaining_content.split("\n\n")
                chunk_content = ""
                used_paragraphs = 0

                for i, paragraph in enumerate(paragraphs):
                    test_content = (
                        chunk_content + "\n\n" + paragraph
                        if chunk_content
                        else paragraph
                    )
                    if len(test_content) <= self.target_chunk_size:
                        chunk_content = test_content
                        used_paragraphs = i + 1
                    else:
                        break

                if used_paragraphs == 0:
                    # 첫 번째 문단도 너무 큰 경우, 강제로 자름
                    chunk_content = remaining_content[: self.target_chunk_size]
                    # 단어 경계에서 자르기
                    last_space = chunk_content.rfind(" ")
                    if last_space > self.target_chunk_size // 2:
                        chunk_content = chunk_content[:last_space]

                    remaining_content = remaining_content[len(chunk_content) :].strip()
                else:
                    # 사용된 문단들 제거
                    remaining_paragraphs = paragraphs[used_paragraphs:]
                    remaining_content = "\n\n".join(remaining_paragraphs).strip()

            # 청크 생성
            if chunk_content.strip():
                chunk = ContentChunk(
                    content=chunk_content.strip(),
                    toc_entry=toc_entry,
                    chunk_index=chunk_index,
                    total_chunks=0,  # 나중에 업데이트
                    section_info={
                        "fallback_chunking": True,
                        "chunk_method": "simple",
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

        # total_chunks 업데이트
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.info(f"폴백 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks

    def _split_paragraph_to_chunks(
        self, paragraph: str, toc_entry: TOCEntry, start_index: int
    ) -> List[ContentChunk]:
        """
        단일 문단을 여러 청크로 분할합니다.

        Args:
            paragraph: 분할할 문단
            toc_entry: TOC 항목
            start_index: 시작 청크 인덱스

        Returns:
            생성된 청크들
        """
        if len(paragraph) <= self.target_chunk_size:
            chunk = ContentChunk(
                content=paragraph,
                toc_entry=toc_entry,
                chunk_index=start_index,
                total_chunks=1,
                section_info={"single_paragraph": True},
            )
            return [chunk]

        # 문장 단위로 분할
        sentences = self._split_by_sentences(paragraph)
        chunks = []
        current_chunk = ""
        chunk_index = start_index

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) <= self.target_chunk_size:
                current_chunk = test_chunk
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunk = ContentChunk(
                        content=current_chunk.strip(),
                        toc_entry=toc_entry,
                        chunk_index=chunk_index,
                        total_chunks=0,  # 나중에 업데이트
                        section_info={
                            "paragraph_split": True,
                            "sentence_based": True,
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # 새 청크 시작
                current_chunk = sentence

        # 마지막 청크 저장
        if current_chunk:
            chunk = ContentChunk(
                content=current_chunk.strip(),
                toc_entry=toc_entry,
                chunk_index=chunk_index,
                total_chunks=0,  # 나중에 업데이트
                section_info={
                    "paragraph_split": True,
                    "sentence_based": True,
                },
            )
            chunks.append(chunk)

        # total_chunks 업데이트
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_by_words(self, text: str, max_size: int) -> List[str]:
        """
        단어 단위로 텍스트를 분할합니다.

        Args:
            text: 분할할 텍스트
            max_size: 최대 크기

        Returns:
            분할된 텍스트 부분들
        """
        if len(text) <= max_size:
            return [text]

        words = text.split()
        parts = []
        current_part = ""

        for word in words:
            test_part = current_part + " " + word if current_part else word

            if len(test_part) <= max_size:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part)
                current_part = word

        if current_part:
            parts.append(current_part)

        return parts

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        텍스트에 대한 임베딩을 생성합니다.

        Args:
            text: 임베딩을 생성할 텍스트

        Returns:
            임베딩 벡터 또는 None
        """
        if not self.gemini_api_key:
            return None

        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            return None

    def save_to_database(self, chunks: List[ContentChunk]):
        """
        청크들을 데이터베이스에 저장합니다.

        Args:
            chunks: 저장할 청크들
        """
        if not chunks:
            logger.warning("저장할 청크가 없습니다.")
            return

        try:
            # 데이터베이스 연결
            conn = psycopg.connect(self.connection_string)
            register_vector(conn)

            # 테이블 존재 확인 및 자동 생성
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'documents'
                    );
                """)
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    logger.info(
                        "documents 테이블이 존재하지 않습니다. 자동으로 생성합니다."
                    )
                    conn.close()
                    setup_database(self.connection_string)
                    conn = psycopg.connect(self.connection_string)
                    register_vector(conn)

            logger.info(f"{len(chunks)}개의 청크를 데이터베이스에 저장하는 중...")

            with conn.cursor() as cursor:
                # 청크별로 처리
                for chunk in tqdm(chunks, desc="청크 저장"):
                    # 임베딩 생성
                    embedding = self.generate_embedding(chunk.content)

                    # 계층 구조 생성
                    hierarchy = {
                        "toc_hierarchy": chunk.toc_entry.hierarchy,
                        "toc_title": chunk.toc_entry.title,
                        "toc_level": chunk.toc_entry.level,
                        "file_path": chunk.toc_entry.file_path,
                    }

                    # 메타데이터 생성
                    metadata = {
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "content_length": len(chunk.content),
                        "section_info": chunk.section_info or {},
                    }

                    # 데이터베이스에 삽입
                    insert_query = """
                    INSERT INTO documents (content, embedding, hierarchy, metadata)
                    VALUES (%s, %s, %s, %s)
                    """

                    cursor.execute(
                        insert_query,
                        (
                            chunk.content,
                            embedding,
                            json.dumps(hierarchy, ensure_ascii=False),
                            json.dumps(metadata, ensure_ascii=False),
                        ),
                    )

                # 변경사항 커밋
                conn.commit()

            logger.info("모든 청크가 성공적으로 저장되었습니다.")

        except Exception as e:
            logger.error(f"데이터베이스 저장 중 오류 발생: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def process_epub(self, epub_file_path: str, save_toc_json: bool = True):
        """
        EPUB 파일을 전체적으로 처리합니다.

        Args:
            epub_file_path: EPUB 파일 경로
            save_toc_json: TOC JSON 파일 저장 여부
        """
        logger.info(f"EPUB 파일 처리 시작: {epub_file_path}")

        try:
            # 1. TOC 추출
            toc_entries = self.extract_toc_hierarchy(epub_file_path)

            if save_toc_json:
                # TOC를 JSON으로 저장
                toc_json_path = Path(epub_file_path).with_suffix("_toc.json")
                toc_data = []

                for entry in toc_entries:
                    toc_data.append(
                        {
                            "title": entry.title,
                            "level": entry.level,
                            "file_path": entry.file_path,
                            "hierarchy": entry.hierarchy,
                            "anchor": entry.anchor,
                        }
                    )

                with open(toc_json_path, "w", encoding="utf-8") as f:
                    json.dump(toc_data, f, ensure_ascii=False, indent=2)

                logger.info(f"TOC JSON 파일 저장: {toc_json_path}")

            # 2. EPUB 파일 읽기
            book = epub.read_epub(epub_file_path)

            # 3. 각 TOC 항목별로 콘텐츠 추출 및 청킹
            all_chunks = []

            for toc_entry in toc_entries:
                logger.info(
                    f"TOC 항목 처리 중: {toc_entry.title} (레벨 {toc_entry.level})"
                )

                # 콘텐츠 추출
                content = self.extract_content_from_file(book, toc_entry.file_path)

                if content.strip():
                    # 스마트 청킹
                    chunks = self.create_smart_chunks(content, toc_entry)
                    all_chunks.extend(chunks)

                    logger.info(f"  -> {len(chunks)}개 청크 생성 (총 {len(content)}자)")
                else:
                    logger.warning(f"  -> 콘텐츠가 비어있습니다: {toc_entry.file_path}")

            logger.info(f"총 {len(all_chunks)}개의 청크가 생성되었습니다.")

            # 통계 출력
            if all_chunks:
                chunk_sizes = [len(chunk.content) for chunk in all_chunks]
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                min_size = min(chunk_sizes)
                max_size = max(chunk_sizes)

                logger.info(f"청크 크기 통계:")
                logger.info(f"  - 평균: {avg_size:.0f}자")
                logger.info(f"  - 최소: {min_size:,}자")
                logger.info(f"  - 최대: {max_size:,}자")
                logger.info(
                    f"  - 목표 대비 평균: {avg_size / self.target_chunk_size * 100:.1f}%"
                )

            # 4. 데이터베이스에 저장
            if all_chunks:
                self.save_to_database(all_chunks)
            else:
                logger.warning(
                    "생성된 청크가 없어서 데이터베이스에 저장할 내용이 없습니다."
                )

            logger.info("EPUB 파일 처리 완료!")

        except Exception as e:
            logger.error(f"EPUB 파일 처리 중 오류 발생: {e}")
            raise

    def _force_split_content(self, content: str) -> List[str]:
        """
        크기가 큰 콘텐츠를 강제로 분할합니다.
        문장 경계 → 문단 경계 → 단어 경계 → 강제 절단 순서로 시도합니다.
        """
        if len(content) <= self.max_chunk_size:
            return [content]

        parts = []
        remaining = content
        safety_margin = 200  # 안전 여유분

        while remaining:
            if len(remaining) <= self.max_chunk_size:
                parts.append(remaining)
                break

            # 목표 크기 (여유분 고려)
            target_size = self.max_chunk_size - safety_margin

            # 1단계: 문장 경계에서 자르기
            cut_pos = None
            sentence_endings = [".", "!", "?", "。", "！", "？"]

            for i in range(target_size, max(target_size // 2, 500), -1):
                if i < len(remaining):
                    char = remaining[i]
                    # 문장 부호 + 다음 문자가 공백인 경우만
                    if (
                        char in sentence_endings
                        and i + 1 < len(remaining)
                        and remaining[i + 1] in [" ", "\n", "\t"]
                    ):
                        cut_pos = i + 1
                        break

            # 2단계: 문단 경계에서 자르기
            if cut_pos is None:
                paragraph_breaks = ["\n\n", "\n"]
                for break_pattern in paragraph_breaks:
                    pos = remaining.rfind(break_pattern, 0, target_size)
                    if pos > target_size // 2:  # 너무 짧지 않게
                        cut_pos = pos + len(break_pattern)
                        break

            # 3단계: 단어 경계에서 자르기
            if cut_pos is None:
                for i in range(target_size, max(target_size // 2, 500), -1):
                    if i < len(remaining):
                        char = remaining[i]
                        # 공백이거나 한글/알파벳이 끝나는 지점
                        if char in [" ", "\n", "\t"]:
                            cut_pos = i
                            break
                        elif i > 0:
                            prev_char = remaining[i - 1]
                            # 한글에서 한글이 아닌 문자로 변하는 지점
                            if ord("가") <= ord(prev_char) <= ord("힣") and not (
                                ord("가") <= ord(char) <= ord("힣")
                            ):
                                cut_pos = i
                                break
                            # 알파벳에서 알파벳이 아닌 문자로 변하는 지점
                            elif prev_char.isalpha() and not char.isalpha():
                                cut_pos = i
                                break

            # 4단계: 강제 절단 (최소 크기 보장)
            if cut_pos is None:
                cut_pos = max(target_size, 1000)  # 최소 1000자는 보장

            # 분할 실행
            current_part = remaining[:cut_pos].strip()
            remaining = remaining[cut_pos:].strip()

            if current_part:
                parts.append(current_part)

            # 무한루프 방지
            if len(remaining) >= len(content):
                logger.error("강제 분할에서 무한루프 감지")
                break

        return parts
