"""
EPUB 파일 분석 및 PostgreSQL 저장 시스템
계층적 TOC 구조를 추출하고 Gemini 임베딩을 생성하여 데이터베이스에 저장합니다.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import psycopg
from pgvector.psycopg import register_vector
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TOCEntry:
    """TOC 항목을 나타내는 데이터 클래스"""

    title: str
    level: int
    file_path: str
    hierarchy: List[Dict[str, Any]]
    anchor: Optional[str] = None


@dataclass
class ContentChunk:
    """콘텐츠 청크를 나타내는 데이터 클래스"""

    content: str
    toc_entry: TOCEntry
    chunk_index: int
    total_chunks: int
    section_info: Optional[Dict[str, Any]] = None


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
            content = item.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")

            soup = BeautifulSoup(content, "html.parser")

            # 스크립트 및 스타일 태그 제거
            for script in soup(["script", "style"]):
                script.decompose()

            # 텍스트 추출
            text = soup.get_text()

            # 텍스트 정리
            # 여러 줄바꿈을 두 줄바꿈으로
            text = re.sub(r"\n\s*\n", "\n\n", text)
            # 여러 공백을 하나의 공백으로
            text = re.sub(r" +", " ", text)
            # 탭을 공백으로
            text = re.sub(r"\t", " ", text)

            text = text.strip()
            logger.debug(f"추출된 텍스트 길이: {len(text)} 문자")

            return text

        except Exception as e:
            logger.error(f"콘텐츠 추출 중 오류 발생 (파일: {file_path}): {e}")
            return ""

    def create_smart_chunks(
        self, content: str, toc_entry: TOCEntry
    ) -> List[ContentChunk]:
        """
        자연스러운 소제목을 감지하여 지능적으로 청크를 분할합니다.

        Args:
            content: 분할할 텍스트 콘텐츠
            toc_entry: TOC 항목 정보

        Returns:
            생성된 청크 리스트
        """
        if not content or not content.strip():
            logger.debug(f"빈 콘텐츠로 인해 청크 생성 건너뜀: {toc_entry.title}")
            return []

        content = content.strip()
        logger.debug(f"스마트 청크 생성 시작: {toc_entry.title} (길이: {len(content)})")

        # 1단계: 자연스러운 소제목 감지
        sections = self._detect_natural_subtitles(content)
        logger.debug(f"감지된 섹션 수: {len(sections)}")

        if len(sections) <= 1:
            # 소제목이 없으면 기본 분할 사용
            return self._create_simple_chunks_fallback(content, toc_entry)

        # 2단계: 섹션을 기반으로 청크 생성
        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._create_chunks_from_section(
                section, toc_entry, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # total_chunks 업데이트
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.debug(f"스마트 청크 생성 완료: {len(chunks)}개 청크")
        if chunks:
            sizes = [len(chunk.content) for chunk in chunks]
            logger.debug(
                f"청크 크기: 평균={sum(sizes) / len(sizes):.0f}, 최소={min(sizes)}, 최대={max(sizes)}"
            )

        return chunks

    def _detect_natural_subtitles(self, content: str) -> List[Dict[str, Any]]:
        """
        자연스러운 소제목을 감지합니다.

        Args:
            content: 분석할 텍스트

        Returns:
            섹션 정보 리스트 [{"title": str, "content": str, "start": int, "end": int}]
        """
        paragraphs = content.split("\n\n")
        sections = []
        current_section = {"title": None, "content": "", "start": 0, "end": 0}

        char_pos = 0

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                char_pos += 2  # \n\n
                continue

            # 소제목 패턴 감지
            if self._is_natural_subtitle(paragraph, i, paragraphs):
                # 이전 섹션 완료
                if current_section["content"].strip():
                    current_section["end"] = char_pos
                    sections.append(current_section.copy())

                # 새 섹션 시작
                current_section = {
                    "title": paragraph,
                    "content": "",
                    "start": char_pos,
                    "end": 0,
                }
                logger.debug(f"소제목 감지: '{paragraph[:50]}...'")
            else:
                # 현재 섹션에 내용 추가
                if current_section["content"]:
                    current_section["content"] += "\n\n" + paragraph
                else:
                    current_section["content"] = paragraph

            char_pos += len(paragraph) + 2  # paragraph + \n\n

        # 마지막 섹션 처리
        if current_section["content"].strip():
            current_section["end"] = len(content)
            sections.append(current_section)

        # 첫 번째 섹션에 제목이 없으면 기본 제목 추가
        if sections and not sections[0]["title"]:
            sections[0]["title"] = "서론"

        return sections

    def _is_natural_subtitle(
        self, paragraph: str, index: int, all_paragraphs: List[str]
    ) -> bool:
        """
        문단이 자연스러운 소제목인지 판단합니다.

        Args:
            paragraph: 검사할 문단
            index: 문단의 인덱스
            all_paragraphs: 전체 문단 리스트

        Returns:
            소제목 여부
        """
        if not paragraph or len(paragraph) > 200:
            return False

        # 1. 길이 기반 판단 (짧고 명확한 제목)
        if len(paragraph) < 80 and ":" in paragraph:
            # "제목: 부제목" 형태
            return True

        # 2. 숫자로 시작하는 경우
        if re.match(r"^\d+[\.\s]", paragraph):
            return True

        # 3. 특정 키워드로 시작하는 경우
        title_keywords = [
            "가장",
            "가장 작은",
            "가장 큰",
            "최초의",
            "마지막",
            "새로운",
            "고전적인",
            "현대적인",
            "전통적인",
            "The",
            "A New",
            "The First",
            "The Last",
        ]
        if any(paragraph.startswith(keyword) for keyword in title_keywords):
            return True

        # 4. 제목 같은 구조 (짧고 다음 문단이 설명적)
        if (
            len(paragraph) < 100
            and not paragraph.endswith(".")
            and index + 1 < len(all_paragraphs)
        ):
            next_paragraph = all_paragraphs[index + 1].strip()
            if len(next_paragraph) > len(paragraph) * 2:  # 다음 문단이 훨씬 긴 경우
                return True

        # 5. 볼드체나 강조 표시 (HTML에서 변환된 경우)
        if (paragraph.isupper() and len(paragraph.split()) <= 8) or (
            paragraph.count(":") == 1 and not paragraph.endswith(".")
        ):
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
            생성된 청크 리스트
        """
        chunks = []

        # 섹션 제목 + 내용 결합
        section_title = section.get("title", "")
        section_content = section.get("content", "")

        if section_title:
            full_content = f"### {section_title}\n\n{section_content}"
        else:
            full_content = section_content

        # 크기가 허용 범위 내이면 단일 청크로
        if len(full_content) <= self.max_chunk_size:
            if len(full_content) >= self.min_chunk_size:
                chunks.append(
                    ContentChunk(
                        content=full_content,
                        toc_entry=toc_entry,
                        chunk_index=start_index,
                        total_chunks=0,
                        section_info={
                            "section_title": section_title,
                            "is_complete_section": True,
                        },
                    )
                )
            return chunks

        # 크기가 큰 경우 분할
        # 제목은 유지하고 내용만 분할
        logger.debug(f"큰 섹션 분할: '{section_title}' (크기: {len(full_content)})")
        content_parts = self._split_content_preserving_paragraphs(section_content)

        for i, part in enumerate(content_parts):
            if section_title and i == 0:
                chunk_content = f"### {section_title}\n\n{part}"
            else:
                chunk_content = part

            if len(chunk_content) >= self.min_chunk_size:
                # 크기 재검증 및 강제 분할
                if len(chunk_content) > self.max_chunk_size:
                    logger.warning(
                        f"청크가 최대 크기 초과: {len(chunk_content)} > {self.max_chunk_size}"
                    )
                    # 강제 분할
                    forced_parts = self._force_split_content(chunk_content)
                    for j, forced_part in enumerate(forced_parts):
                        chunks.append(
                            ContentChunk(
                                content=forced_part,
                                toc_entry=toc_entry,
                                chunk_index=start_index + len(chunks),
                                total_chunks=0,
                                section_info={
                                    "section_title": section_title,
                                    "is_complete_section": False,
                                    "part_index": i,
                                    "total_parts": len(content_parts),
                                    "forced_split": True,
                                    "forced_part": j,
                                },
                            )
                        )
                else:
                    chunks.append(
                        ContentChunk(
                            content=chunk_content,
                            toc_entry=toc_entry,
                            chunk_index=start_index + len(chunks),
                            total_chunks=0,
                            section_info={
                                "section_title": section_title,
                                "is_complete_section": len(content_parts) == 1,
                                "part_index": i,
                                "total_parts": len(content_parts),
                            },
                        )
                    )

        return chunks

    def _split_content_preserving_paragraphs(self, content: str) -> List[str]:
        """
        문단 경계를 보존하면서 내용을 분할합니다.

        Args:
            content: 분할할 내용

        Returns:
            분할된 내용 리스트
        """
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        parts = []
        current_part = ""

        for paragraph in paragraphs:
            test_content = (
                current_part + "\n\n" + paragraph if current_part else paragraph
            )

            if len(test_content) <= self.max_chunk_size:
                current_part = test_content
            else:
                if current_part:
                    parts.append(current_part)
                    current_part = paragraph
                else:
                    # 단일 문단이 너무 긴 경우 강제 분할
                    sentence_parts = self._split_by_sentences(paragraph)
                    parts.extend(sentence_parts)

        if current_part:
            parts.append(current_part)

        return parts

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        문장 단위로 텍스트를 분할합니다.

        Args:
            text: 분할할 텍스트

        Returns:
            분할된 텍스트 리스트
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        parts = []
        current_part = ""

        for sentence in sentences:
            test_part = current_part + " " + sentence if current_part else sentence

            if len(test_part) <= self.max_chunk_size:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part)
                current_part = sentence

        if current_part:
            parts.append(current_part)

        return parts

    def _create_simple_chunks_fallback(
        self, content: str, toc_entry: TOCEntry
    ) -> List[ContentChunk]:
        """
        소제목이 감지되지 않을 때 사용하는 기본 청킹 방법입니다.

        Args:
            content: 분할할 텍스트 콘텐츠
            toc_entry: TOC 항목 정보

        Returns:
            생성된 청크 리스트
        """
        if not content or not content.strip():
            logger.debug(f"빈 콘텐츠로 인해 청크 생성 건너뜀: {toc_entry.title}")
            return []

        content = content.strip()
        logger.debug(f"청크 생성 시작: {toc_entry.title} (길이: {len(content)})")

        # 기본 분할: 문단 단위
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            paragraphs = [content]

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            # 현재 청크에 문단을 추가했을 때의 크기 확인
            test_chunk = (
                current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            )

            if len(test_chunk) <= self.max_chunk_size:
                # 크기가 허용 범위 내면 추가
                current_chunk = test_chunk
            else:
                # 현재 청크가 있으면 먼저 저장
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(
                        ContentChunk(
                            content=current_chunk,
                            toc_entry=toc_entry,
                            chunk_index=chunk_index,
                            total_chunks=0,  # 나중에 업데이트
                        )
                    )
                    chunk_index += 1
                    current_chunk = ""

                # 단일 문단이 최대 크기를 초과하는 경우 문장 단위로 분할
                if len(paragraph) > self.max_chunk_size:
                    sentence_chunks = self._split_paragraph_to_chunks(
                        paragraph, toc_entry, chunk_index
                    )
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                else:
                    current_chunk = paragraph

        # 마지막 청크 추가
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(
                ContentChunk(
                    content=current_chunk,
                    toc_entry=toc_entry,
                    chunk_index=chunk_index,
                    total_chunks=0,
                )
            )

        # total_chunks 업데이트
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.debug(f"청크 생성 완료: {len(chunks)}개 청크")
        if chunks:
            sizes = [len(chunk.content) for chunk in chunks]
            logger.debug(
                f"청크 크기: 평균={sum(sizes) / len(sizes):.0f}, 최소={min(sizes)}, 최대={max(sizes)}"
            )

        return chunks

    def _split_paragraph_to_chunks(
        self, paragraph: str, toc_entry: TOCEntry, start_index: int
    ) -> List[ContentChunk]:
        """
        큰 문단을 문장 단위로 청크로 분할합니다.

        Args:
            paragraph: 분할할 문단
            toc_entry: TOC 항목
            start_index: 시작 인덱스

        Returns:
            생성된 청크 리스트
        """
        # 문장 분할 (간단한 정규식 사용)
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        chunk_index = start_index

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) <= self.max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(
                        ContentChunk(
                            content=current_chunk,
                            toc_entry=toc_entry,
                            chunk_index=chunk_index,
                            total_chunks=0,
                        )
                    )
                    chunk_index += 1

                # 단일 문장이 너무 긴 경우 강제 분할
                if len(sentence) > self.max_chunk_size:
                    word_chunks = self._split_by_words(sentence, self.max_chunk_size)
                    for word_chunk in word_chunks:
                        chunks.append(
                            ContentChunk(
                                content=word_chunk,
                                toc_entry=toc_entry,
                                chunk_index=chunk_index,
                                total_chunks=0,
                            )
                        )
                        chunk_index += 1
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(
                ContentChunk(
                    content=current_chunk,
                    toc_entry=toc_entry,
                    chunk_index=chunk_index,
                    total_chunks=0,
                )
            )

        return chunks

    def _split_by_words(self, text: str, max_size: int) -> List[str]:
        """
        텍스트를 단어 단위로 분할합니다.

        Args:
            text: 분할할 텍스트
            max_size: 최대 크기

        Returns:
            분할된 텍스트 조각들
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # 공백 포함

            if current_size + word_size <= max_size:
                current_chunk.append(word)
                current_size += word_size
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Gemini를 사용하여 텍스트의 임베딩을 생성합니다.

        Args:
            text: 임베딩을 생성할 텍스트

        Returns:
            임베딩 벡터 또는 None (실패 시)
        """
        if not self.gemini_api_key:
            return None

        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="semantic_similarity",
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            return None

    def save_to_database(self, chunks: List[ContentChunk]):
        """
        청크들을 데이터베이스에 저장합니다.

        Args:
            chunks: 저장할 청크 리스트
        """
        if not chunks:
            logger.warning("저장할 청크가 없습니다.")
            return

        logger.info(f"데이터베이스에 {len(chunks)}개 청크 저장 시작")

        try:
            conn = psycopg.connect(self.connection_string)
            register_vector(conn)
            cursor = conn.cursor()

            # 테이블 존재 여부 확인 및 생성
            try:
                cursor.execute("SELECT COUNT(*) FROM documents LIMIT 1;")
            except Exception as e:
                if "does not exist" in str(e):
                    logger.warning(
                        "documents 테이블이 없습니다. 데이터베이스 설정을 실행합니다."
                    )
                    cursor.close()
                    conn.close()

                    # 데이터베이스 설정 실행
                    from database_setup import setup_database

                    setup_database(self.connection_string)

                    # 다시 연결
                    conn = psycopg.connect(self.connection_string)
                    register_vector(conn)
                    cursor = conn.cursor()
                else:
                    raise

            insert_count = 0
            for chunk in tqdm(chunks, desc="데이터베이스 저장"):
                try:
                    # 임베딩 생성
                    embedding = self.generate_embedding(chunk.content)

                    # 메타데이터 구성
                    metadata = {
                        "title": chunk.toc_entry.title,
                        "file_path": chunk.toc_entry.file_path,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "toc_level": chunk.toc_entry.level,
                        "anchor": chunk.toc_entry.anchor,
                        "chunk_size": len(chunk.content),
                    }

                    # 문서 저장
                    cursor.execute(
                        """
                        INSERT INTO documents (
                            content, embedding, hierarchy, metadata
                        ) VALUES (%s, %s, %s, %s)
                        """,
                        (
                            chunk.content,
                            embedding,
                            json.dumps(chunk.toc_entry.hierarchy, ensure_ascii=False),
                            json.dumps(metadata, ensure_ascii=False),
                        ),
                    )
                    insert_count += 1

                except Exception as e:
                    logger.error(f"청크 저장 중 오류 발생: {e}")
                    continue

            conn.commit()
            logger.info(f"데이터베이스에 {insert_count}개 청크 저장 완료")

        except Exception as e:
            logger.error(f"데이터베이스 저장 중 오류 발생: {e}")
            raise

    def process_epub(self, epub_file_path: str, save_toc_json: bool = True):
        """
        EPUB 파일을 전체적으로 처리합니다.

        Args:
            epub_file_path: 처리할 EPUB 파일 경로
            save_toc_json: TOC를 JSON 파일로 저장할지 여부
        """
        logger.info(f"EPUB 파일 처리 시작: {epub_file_path}")

        if not os.path.exists(epub_file_path):
            raise FileNotFoundError(f"EPUB 파일을 찾을 수 없습니다: {epub_file_path}")

        # 1. TOC 추출
        logger.info("1단계: TOC 추출 중...")
        toc_entries = self.extract_toc_hierarchy(epub_file_path)

        if save_toc_json:
            toc_json_path = epub_file_path.replace(".epub", "_toc.json")
            with open(toc_json_path, "w", encoding="utf-8") as f:
                toc_data = [
                    {
                        "title": entry.title,
                        "level": entry.level,
                        "file_path": entry.file_path,
                        "hierarchy": entry.hierarchy,
                        "anchor": entry.anchor,
                    }
                    for entry in toc_entries
                ]
                json.dump(toc_data, f, ensure_ascii=False, indent=2)
            logger.info(f"TOC가 저장되었습니다: {toc_json_path}")

        # 2. 콘텐츠 추출 및 청크 생성
        logger.info("2단계: 콘텐츠 추출 및 청킹 중...")
        book = epub.read_epub(epub_file_path)
        all_chunks = []

        # 파일별로 처리하여 중복 방지
        processed_files = set()

        for i, toc_entry in enumerate(toc_entries):
            logger.debug(f"처리 중 ({i + 1}/{len(toc_entries)}): {toc_entry.title}")

            if toc_entry.file_path in processed_files:
                logger.debug(f"이미 처리된 파일 건너뜀: {toc_entry.file_path}")
                continue

            processed_files.add(toc_entry.file_path)

            # 콘텐츠 추출
            content = self.extract_content_from_file(book, toc_entry.file_path)

            if content:
                # 스마트 청킹 (자연스러운 소제목 감지)
                chunks = self.create_smart_chunks(content, toc_entry)
                all_chunks.extend(chunks)
                logger.debug(
                    f"{toc_entry.title}: {len(chunks)}개 청크 생성 (내용 길이: {len(content)})"
                )
            else:
                logger.debug(f"{toc_entry.title}: 빈 콘텐츠로 건너뜀")

        logger.info(f"총 {len(all_chunks)}개의 콘텐츠 청크가 생성되었습니다.")

        # 청크 크기 통계
        if all_chunks:
            chunk_sizes = [len(chunk.content) for chunk in all_chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)

            logger.info(
                f"청크 크기 통계: 평균 {avg_size:.0f}자, 최소 {min_size}자, 최대 {max_size}자"
            )

            # 목표 대비 분석
            target_ratio = (avg_size / self.target_chunk_size) * 100
            logger.info(f"목표 크기 대비: {target_ratio:.1f}%")

        # 3. 데이터베이스에 저장
        if all_chunks:
            logger.info("3단계: 데이터베이스 저장 중...")
            self.save_to_database(all_chunks)
        else:
            logger.warning("저장할 콘텐츠 청크가 없습니다.")

        logger.info("EPUB 파일 처리가 완료되었습니다.")

    def _force_split_content(self, content: str) -> List[str]:
        """
        크기가 초과된 콘텐츠를 강제로 분할합니다.
        문장 -> 단어 -> 문자 순서로 경계를 찾아서 분할합니다.

        Args:
            content: 분할할 콘텐츠

        Returns:
            분할된 콘텐츠 리스트
        """
        if len(content) <= self.max_chunk_size:
            return [content]

        parts = []
        current_pos = 0

        while current_pos < len(content):
            # 안전 여유분을 두고 자르기 (200자 여유)
            target_end = current_pos + self.max_chunk_size - 200

            if target_end >= len(content):
                # 마지막 부분
                parts.append(content[current_pos:])
                break

            # 현재 위치에서 목표 지점까지의 텍스트
            search_text = content[current_pos : target_end + 200]  # 여유분 포함 검색
            best_cut_pos = target_end - current_pos  # 기본값

            # 1단계: 문장 경계 찾기 (뒤에서부터)
            sentence_endings = []
            for i, char in enumerate(search_text):
                if char in ".!?。！？":
                    # 다음 문자가 공백이나 줄바꿈인지 확인
                    if i + 1 < len(search_text) and search_text[i + 1] in " \n\t":
                        sentence_endings.append(i + 1)  # 문장 부호 다음 위치

            # 적절한 문장 끝 찾기 (목표 크기에 가장 가까운 것)
            for pos in reversed(sentence_endings):
                if (
                    pos <= target_end - current_pos
                    and pos >= (target_end - current_pos) // 2
                ):
                    best_cut_pos = pos
                    break

            # 2단계: 문장 경계를 못 찾았으면 문단 경계 찾기
            if best_cut_pos == target_end - current_pos:
                paragraph_breaks = []
                for i in range(len(search_text) - 1):
                    if search_text[i : i + 2] == "\n\n":
                        paragraph_breaks.append(i)

                for pos in reversed(paragraph_breaks):
                    if (
                        pos <= target_end - current_pos
                        and pos >= (target_end - current_pos) // 2
                    ):
                        best_cut_pos = pos
                        break

            # 3단계: 단어 경계 찾기
            if best_cut_pos == target_end - current_pos:
                # 목표 지점에서 뒤로 가면서 공백 찾기
                for i in range(
                    min(target_end - current_pos, len(search_text)) - 1,
                    max(0, (target_end - current_pos) // 2),
                    -1,
                ):
                    if search_text[i] in " \n\t\r":
                        # 단어 경계 확인: 앞뒤 모두 알파벳이나 한글이 아닌 경우
                        if i > 0 and i < len(search_text) - 1:
                            prev_char = search_text[i - 1]
                            next_char = search_text[i + 1]
                            if not (prev_char.isalnum() and next_char.isalnum()):
                                best_cut_pos = i
                                break
                        else:
                            best_cut_pos = i
                            break

            # 4단계: 모든 경계를 못 찾았으면 강제로 자르되, 최소 길이 보장
            if best_cut_pos == target_end - current_pos:
                # 최소 청크 크기의 2배는 되도록 보장
                min_safe_size = max(self.min_chunk_size * 2, 1000)
                if target_end - current_pos < min_safe_size:
                    best_cut_pos = min(min_safe_size, len(content) - current_pos)
                else:
                    best_cut_pos = target_end - current_pos

            # 청크 추가
            chunk_text = content[current_pos : current_pos + best_cut_pos].strip()
            if chunk_text:
                parts.append(chunk_text)

            # 다음 위치로 이동 (공백 건너뛰기)
            current_pos += best_cut_pos
            while current_pos < len(content) and content[current_pos] in " \n\t\r":
                current_pos += 1

        # 빈 부분 제거
        parts = [part.strip() for part in parts if part.strip()]

        logger.debug(
            f"강제 분할: {len(content)}자 -> {len(parts)}개 조각 (크기: {[len(p) for p in parts]})"
        )
        return parts


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="EPUB 파일을 분석하고 PostgreSQL에 저장합니다."
    )
    parser.add_argument("epub_file", help="처리할 EPUB 파일 경로")
    parser.add_argument(
        "--db-url",
        default="postgresql://user@localhost:5432/postgres",
        help="PostgreSQL 연결 URL",
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=4096,
        help="목표 청크 크기 (기본값: 4096)",
    )
    parser.add_argument(
        "--min-chunk-size", type=int, default=1000, help="최소 청크 크기 (기본값: 1000)"
    )
    parser.add_argument(
        "--max-chunk-size", type=int, default=6000, help="최대 청크 크기 (기본값: 6000)"
    )
    parser.add_argument(
        "--no-toc-json", action="store_true", help="TOC JSON 파일 저장 건너뛰기"
    )

    args = parser.parse_args()

    try:
        processor = EPUBProcessor(
            connection_string=args.db_url,
            target_chunk_size=args.target_chunk_size,
            min_chunk_size=args.min_chunk_size,
            max_chunk_size=args.max_chunk_size,
        )

        processor.process_epub(
            epub_file_path=args.epub_file, save_toc_json=not args.no_toc_json
        )

    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
