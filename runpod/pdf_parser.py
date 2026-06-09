from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


"""
MinerU PDF parser for RAG preprocessing.

Design goals:
1. Original PDF filenames never go into MinerU CLI.
   - Every PDF is copied to an ASCII-only temporary path as input.pdf.
   - This avoids Korean names, spaces, brackets, long names, and special characters.
2. Final outputs are flat, not PDF-per-folder.
   - parsed/markdown/doc_<hash>.md
   - parsed/json/doc_<hash>.content_list.json
   - parsed/json/doc_<hash>.middle.json
   - parsed/json/all_blocks.jsonl
   - parsed/assets/doc_<hash>__asset_000001.ext
   - parsed/text/doc_<hash>.pymupdf.txt
   - parsed/reports/doc_<hash>.quality_report.json
3. Original filenames are preserved only in JSON metadata and manifests.
4. Parsing only. Chunking/embedding/vector DB ingestion should be separate modules.
5. Adds quality metadata for RAG preprocessing:
   - article-number validation
   - appendix/form region labeling
   - suspicious OCR/table flags
   - optional PyMuPDF text extraction for validation fallback
"""


AUXILIARY_BLOCK_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
}

IMAGE_PATH_FIELDS = {
    "img_path",
    "image_path",
    "table_img_path",
    "chart_img_path",
}

TEXT_FIELDS_IN_PRIORITY = (
    "text",
    "content",
    "table_caption",
    "table_body",
    "table_footnote",
    "image_caption",
    "image_footnote",
    "chart_caption",
    "chart_footnote",
    "equation",
    "latex",
    "code_caption",
    "code_body",
)

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
PAGE_NUMBER_LINE_PATTERNS = [
    re.compile(r"^\s*[-–—]?\s*\d{1,5}\s*[-–—]?\s*$"),
    re.compile(r"^\s*(?:page|p\.?)\s*\d{1,5}\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:page|p\.?)?\s*\d{1,5}\s*(?:/|of)\s*\d{1,5}\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,5}\s*/\s*\d{1,5}\s*$"),
]

# Korean regulation / rule documents often have article headings such as
# "제20조(입찰참가신청)". These patterns are used only for validation metadata;
# they do not change the original parser output.
ARTICLE_RE = re.compile(r"제\s*(\d{1,4})\s*조(?:의\s*(\d{1,3}))?\s*(?:\(([^\n\)]{1,80})\))?")
APPENDIX_RE = re.compile(r"\[?별지\s*제?\s*\d+[^\n\]]*서식\]?|별지\s*제?\s*\d+[^\n]*서식")
SUPPLEMENTARY_RE = re.compile(r"^\s*부\s*칙", re.MULTILINE)

# Conservative corrections for common OCR artifacts observed in the supplied
# MinerU output. These are intentionally narrow and are applied only when
# --enable-ocr-corrections is active.
KNOWN_OCR_CORRECTIONS: list[tuple[str, str]] = [
    ("➂", "③"),
    ("2122.12.21", "2022.12.21"),
    ("2025.12.일26", "2025.12.26"),
    ("국가를 당사자로 하는 계약에 관한 법률J", "국가를 당사자로 하는 계약에 관한 법률」"),
    ("댓표잣주 소", "대표자 주소"),
    ("살업장주부소", "사업장 주소"),
    ("살업작등록번호", "사업자등록번호"),
    ("전념도매훌액", "전년도 매출액"),
    ("경화파사종합판정", "검사 종합판정"),
    ("경하이사종합의견", "검사 종합의견"),
    ("O들루가O참가범위", "등록기간: 참가범위"),
    ("H고", "비고"),
]

SUSPICIOUS_OCR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("template_placeholder_or_garbled_token", re.compile(r"@d-@N|O들루가O|\bz\b")),
    ("broken_korean_form_field", re.compile(r"댓표잣|살업장|살업작|전념도매|경화파사|경하이사")),
    ("broken_legal_bracket", re.compile(r"법률J|시행령J|시행규칙J")),
    ("broken_date_token", re.compile(r"\b(?:19|20|21)\d{2}\.\d{1,2}\.(?:일)?\d{1,2}\b")),
    ("circled_number_variant", re.compile(r"➂")),
]


class ParserError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse PDFs in pdf_data using MinerU. Original filenames are never passed "
            "to MinerU; safe temporary names are used instead."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pdf_data"),
        help="PDF input directory. Default: pdf_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("parsed"),
        help="Final parsing output directory. Default: parsed",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "ASCII-only working directory used for MinerU input/output. "
            "Default: auto-selects a short ASCII path such as /tmp/mineru_work or C:\\mineru_work."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hybrid-auto-engine",
        choices=[
            "pipeline",
            "hybrid-auto-engine",
            "hybrid-http-client",
            "vlm-auto-engine",
            "vlm-http-client",
        ],
        help=(
            "MinerU backend. RunPod GPU default is hybrid-auto-engine. "
            "Use pipeline only for CPU fallback/debugging."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="MinerU parse method. Default: auto",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="korean",
        help="OCR language hint. Use 'none' to omit language option. Default: korean",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search PDFs recursively under input directory.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for batch processing after sorting PDFs. Default: 0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to process. Useful for RunPod/GPU benchmark runs.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Process only PDF filenames containing this substring.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse even when final outputs already exist.",
    )
    parser.add_argument(
        "--auto-fallback",
        action="store_true",
        default=True,
        help="Retry with safer options when the first MinerU attempt fails. Default: enabled.",
    )
    parser.add_argument(
        "--no-auto-fallback",
        dest="auto_fallback",
        action="store_false",
        help="Disable automatic fallback attempts.",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep temporary MinerU working directories for debugging.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Copy raw MinerU output tree into parsed/raw_outputs/doc_<hash>.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout per MinerU attempt in seconds. 0 means no timeout.",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help='Set CUDA_VISIBLE_DEVICES for the MinerU subprocess. Default: "0" on RunPod. Example: --cuda "0" or --cuda "0,1".',
    )
    parser.add_argument(
        "--require-cuda",
        dest="require_cuda",
        action="store_true",
        default=True,
        help="Fail before parsing if PyTorch cannot see a CUDA device. Default: enabled for RunPod GPU.",
    )
    parser.add_argument(
        "--no-require-cuda",
        dest="require_cuda",
        action="store_false",
        help="Allow running without CUDA. Use only for CPU fallback/debugging.",
    )
    parser.add_argument(
        "--formula",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable formula parsing. Default: false for faster RunPod parsing",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable table parsing. Default: true",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="MinerU API URL for http-client backends, if needed.",
    )
    parser.add_argument(
        "--disable-repeated-margin-clean",
        action="store_true",
        help="Disable repeated margin text cleanup. Auxiliary block-type cleanup remains enabled.",
    )
    parser.add_argument(
        "--enable-ocr-corrections",
        dest="enable_ocr_corrections",
        action="store_true",
        default=True,
        help="Apply narrow OCR corrections for known parsing artifacts. Default: enabled.",
    )
    parser.add_argument(
        "--disable-ocr-corrections",
        dest="enable_ocr_corrections",
        action="store_false",
        help="Disable OCR correction replacements and only flag suspicious text.",
    )
    parser.add_argument(
        "--secondary-text",
        dest="secondary_text",
        action="store_true",
        default=True,
        help="Export fast PyMuPDF text for validation and article-number comparison. Default: enabled.",
    )
    parser.add_argument(
        "--no-secondary-text",
        dest="secondary_text",
        action="store_false",
        help="Disable PyMuPDF secondary text extraction.",
    )
    parser.add_argument(
        "--quality-report",
        dest="quality_report",
        action="store_true",
        default=True,
        help="Write parsed/reports/doc_<hash>.quality_report.json. Default: enabled.",
    )
    parser.add_argument(
        "--no-quality-report",
        dest="quality_report",
        action="store_false",
        help="Disable quality report generation.",
    )

    return parser.parse_args()


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def can_write_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def select_work_dir(user_work_dir: Path | None) -> Path:
    if user_work_dir is not None:
        work_dir = user_work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    env_work = os.environ.get("MINERU_WORK_DIR")
    candidates: list[Path] = []

    if env_work:
        candidates.append(Path(env_work))

    if platform.system().lower().startswith("win"):
        system_drive = os.environ.get("SystemDrive", "C:")
        candidates.extend([
            Path(system_drive + "\\mineru_work"),
            Path("C:\\Temp\\mineru_work"),
        ])
    else:
        candidates.extend([
            Path("/tmp/mineru_work"),
            Path("/var/tmp/mineru_work"),
        ])

    # Final fallback. This may contain non-ASCII if the project path does, but it is better than failing.
    candidates.append(Path(".mineru_work"))

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if not is_ascii_path(candidate):
            continue

        if can_write_dir(candidate):
            return candidate

    fallback = Path(".mineru_work").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "markdown": output_dir / "markdown",
        "json": output_dir / "json",
        "assets": output_dir / "assets",
        "text": output_dir / "text",
        "reports": output_dir / "reports",
        "logs": output_dir / "logs",
        "manifests": output_dir / "manifests",
        "raw": output_dir / "raw_outputs",
    }

    for key, path in dirs.items():
        if key == "raw":
            continue
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def check_mineru_cli() -> None:
    if shutil.which("mineru") is None:
        raise ParserError(
            "mineru CLI was not found. Install it first:\n"
            "  pip install --upgrade pip\n"
            "  pip install uv\n"
            "  uv pip install -U \"mineru[all]\"\n"
            "Then check:\n"
            "  mineru --help"
        )


def find_pdfs(input_dir: Path, recursive: bool, only: str | None = None, start_index: int = 0, limit: int | None = None) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(
        path for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if only:
        pdfs = [path for path in pdfs if only in path.name]
    if start_index:
        pdfs = pdfs[start_index:]
    if limit is not None:
        pdfs = pdfs[:limit]
    return pdfs


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_doc_id(pdf_path: Path) -> tuple[str, str]:
    sha = file_sha256(pdf_path)
    return f"doc_{sha[:16]}", sha


def is_page_number_line(line: str) -> bool:
    value = line.strip()
    if not value:
        return False
    return any(pattern.match(value) for pattern in PAGE_NUMBER_LINE_PATTERNS)


def normalize_text(text: str, preserve_newlines: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_RE.sub("", text)

    lines: list[str] = []
    for line in text.split("\n"):
        line = MULTI_SPACE_RE.sub(" ", line).strip()
        if is_page_number_line(line):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = MULTI_BLANK_LINE_RE.sub("\n\n", cleaned).strip()

    if not preserve_newlines:
        cleaned = MULTI_SPACE_RE.sub(" ", cleaned.replace("\n", " ")).strip()

    return cleaned


def normalized_key(text: str) -> str:
    value = normalize_text(text, preserve_newlines=False)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def correct_ocr_text(text: str) -> str:
    """Apply very conservative OCR replacements.

    The intent is not to rewrite legal content aggressively.  These mappings
    target deterministic artifacts observed in generated forms and date tokens.
    """
    if not isinstance(text, str) or not text:
        return ""

    corrected = text
    for old, new in KNOWN_OCR_CORRECTIONS:
        corrected = corrected.replace(old, new)
    return corrected


def detect_ocr_flags(text: str) -> list[str]:
    if not text:
        return []

    flags: list[str] = []
    for name, pattern in SUSPICIOUS_OCR_PATTERNS:
        if pattern.search(text):
            flags.append(name)
    return sorted(set(flags))


def extract_article_refs(text: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for match in ARTICLE_RE.finditer(text or ""):
        article_no = int(match.group(1))
        sub_no = int(match.group(2)) if match.group(2) else None
        title = normalize_text(match.group(3) or "", preserve_newlines=False) or None
        refs.append({"article_no": article_no, "article_sub_no": sub_no, "article_title": title})
    return refs


def table_empty_cell_ratio(text: str) -> float | None:
    """Approximate empty-cell ratio for HTML-like tables in MinerU markdown/text."""
    if "<td" not in text.lower() and "<th" not in text.lower():
        return None
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", text, flags=re.IGNORECASE | re.DOTALL)
    if not cells:
        return None
    empty = 0
    for cell in cells:
        cell_text = re.sub(r"<[^>]+>", "", cell)
        if not normalize_text(cell_text, preserve_newlines=False):
            empty += 1
    return round(empty / len(cells), 4)


def infer_content_category(block_type: str | None, text: str, current_region: str) -> str:
    if current_region == "appendix_form":
        return "appendix_form"
    if current_region == "supplementary":
        return "supplementary"
    if block_type in {"image", "chart"}:
        return "visual"
    if block_type == "table":
        return "table"
    if ARTICLE_RE.search(text or ""):
        return "article_body"
    return "body"


def enrich_block_quality_metadata(block: dict[str, Any], current_region: str) -> dict[str, Any]:
    text = get_primary_text_from_block(block)
    block_type = block.get("type")
    article_refs = extract_article_refs(text)
    empty_ratio = table_empty_cell_ratio(text)
    flags = detect_ocr_flags(text)

    if empty_ratio is not None and empty_ratio >= 0.35:
        flags.append("many_empty_table_cells")
    if len(text) <= 2 and block_type not in {"image", "chart"}:
        flags.append("very_short_text")
    if re.search(r"제\s*\d{1,4}\s*조\([^\)]*$", text):
        flags.append("possibly_truncated_article_heading")
    if re.search(r"입찰참가신\s*$|입찰참가신청\s*$", text):
        flags.append("possibly_truncated_sentence")

    block["content_category"] = infer_content_category(str(block_type) if block_type else None, text, current_region)
    block["quality_flags"] = sorted(set(flags))
    block["ocr_suspicious"] = bool(flags)
    block["table_empty_cell_ratio"] = empty_ratio

    if article_refs:
        first = article_refs[0]
        block["article_no"] = first.get("article_no")
        block["article_sub_no"] = first.get("article_sub_no")
        block["article_title"] = first.get("article_title")
        block["article_refs"] = article_refs

    return block


def extract_pdf_text_with_pymupdf(pdf_path: Path) -> tuple[str, dict[str, Any]]:
    """Fast secondary text extraction for validation.

    This is not a replacement for MinerU. It is used to detect missing legal
    article headings and to keep a plain-text fallback for born-digital PDFs.
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return "", {"available": False, "reason": f"PyMuPDF import failed: {type(exc).__name__}: {exc}"}

    try:
        doc = fitz.open(str(pdf_path))
        pages: list[str] = []
        for idx, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            page_text = normalize_text(page_text, preserve_newlines=True)
            pages.append(f"<PAGE {idx}>\n{page_text}".strip())
        text = "\n\n".join(pages).strip() + "\n"
        return text, {"available": True, "page_count": len(doc), "char_count": len(text)}
    except Exception as exc:  # noqa: BLE001
        return "", {"available": False, "reason": f"PyMuPDF extraction failed: {type(exc).__name__}: {exc}"}


def article_number_summary(text: str) -> dict[str, Any]:
    article_numbers = sorted({ref["article_no"] for ref in extract_article_refs(text) if ref.get("article_no") is not None})
    if not article_numbers:
        return {"article_numbers": [], "article_count": 0, "missing_simple_articles": []}

    expected = set(range(min(article_numbers), max(article_numbers) + 1))
    missing = sorted(expected - set(article_numbers))
    return {
        "article_numbers": article_numbers,
        "article_count": len(article_numbers),
        "min_article_no": min(article_numbers),
        "max_article_no": max(article_numbers),
        "missing_simple_articles": missing,
    }


def build_quality_report(
    cleaned_content_list: list[dict[str, Any]],
    pdf_text: str,
    pdf_text_info: dict[str, Any],
    doc_id: str,
    source_pdf: str,
    source_filename: str,
) -> dict[str, Any]:
    block_texts = [get_primary_text_from_block(block) for block in cleaned_content_list if isinstance(block, dict)]
    combined_text = "\n\n".join(block_texts)

    block_type_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}

    suspicious_samples: list[dict[str, Any]] = []
    for block in cleaned_content_list:
        block_type = str(block.get("type", "unknown"))
        category = str(block.get("content_category", "unknown"))
        block_type_counts[block_type] = block_type_counts.get(block_type, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        for flag in block.get("quality_flags", []) or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        if block.get("quality_flags") and len(suspicious_samples) < 20:
            suspicious_samples.append(
                {
                    "original_block_index": block.get("original_block_index"),
                    "page_idx": block.get("page_idx"),
                    "page_no": block.get("page_idx") + 1 if isinstance(block.get("page_idx"), int) else None,
                    "type": block.get("type"),
                    "content_category": block.get("content_category"),
                    "quality_flags": block.get("quality_flags"),
                    "text_preview": get_primary_text_from_block(block)[:300],
                }
            )

    mineru_articles = article_number_summary(combined_text)
    pdf_articles = article_number_summary(pdf_text) if pdf_text else {"article_numbers": [], "article_count": 0, "missing_simple_articles": []}

    mineru_article_set = set(mineru_articles.get("article_numbers", []))
    pdf_article_set = set(pdf_articles.get("article_numbers", []))

    return {
        "doc_id": doc_id,
        "source_pdf": source_pdf,
        "source_filename": source_filename,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "block_count": len(cleaned_content_list),
        "block_type_counts": block_type_counts,
        "content_category_counts": category_counts,
        "quality_flag_counts": flag_counts,
        "suspicious_block_count": sum(1 for block in cleaned_content_list if block.get("quality_flags")),
        "suspicious_samples": suspicious_samples,
        "mineru_article_summary": mineru_articles,
        "secondary_pdf_text_info": pdf_text_info,
        "secondary_pdf_article_summary": pdf_articles,
        "articles_present_in_pdf_text_but_missing_in_mineru": sorted(pdf_article_set - mineru_article_set),
        "articles_present_in_mineru_but_missing_in_pdf_text": sorted(mineru_article_set - pdf_article_set),
        "recommended_next_step": "review_suspicious_blocks_before_chunking" if flag_counts or (pdf_article_set - mineru_article_set) else "ready_for_chunking",
    }


def check_cuda_requirement(args: argparse.Namespace) -> None:
    if not args.require_cuda:
        return
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ParserError(f"--require-cuda was set, but torch import failed: {type(exc).__name__}: {exc}") from exc

    if not torch.cuda.is_available():
        raise ParserError("--require-cuda was set, but torch.cuda.is_available() is False.")

    device_name = torch.cuda.get_device_name(0)
    print(f"[INFO] CUDA available: {device_name}")


def get_text_parts_from_block(block: dict[str, Any]) -> list[str]:
    parts: list[str] = []

    for field in TEXT_FIELDS_IN_PRIORITY:
        value = block.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if item is not None)

    list_items = block.get("list_items")
    if isinstance(list_items, list):
        parts.extend(str(item) for item in list_items if item is not None)

    return parts


def get_primary_text_from_block(block: dict[str, Any]) -> str:
    return normalize_text("\n".join(get_text_parts_from_block(block)), preserve_newlines=True)


def block_has_image_path(block: dict[str, Any]) -> bool:
    if not isinstance(block, dict):
        return False
    for key, value in block.items():
        if key in IMAGE_PATH_FIELDS and isinstance(value, str) and value.strip():
            return True
        if isinstance(value, dict) and block_has_image_path(value):
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and block_has_image_path(item):
                    return True
    return False


def extract_bbox(block: dict[str, Any]) -> list[float] | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None

    try:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    except (TypeError, ValueError):
        return None


def block_is_margin_like(block: dict[str, Any]) -> bool:
    bbox = extract_bbox(block)
    if bbox is None:
        return False

    _, y0, _, y1 = bbox

    # MinerU bbox coordinates vary by page scale. This heuristic catches obvious top/bottom bands.
    # It is intentionally conservative and only used with repeated-text detection.
    return y0 <= 120 or y1 >= 900


def detect_repeated_margin_texts(content_list: list[dict[str, Any]]) -> set[str]:
    pages = {
        block.get("page_idx")
        for block in content_list
        if isinstance(block, dict) and isinstance(block.get("page_idx"), int)
    }
    total_pages = len(pages)
    if total_pages < 3:
        return set()

    text_to_pages: dict[str, set[int]] = {}

    for block in content_list:
        if not isinstance(block, dict):
            continue
        if not block_is_margin_like(block):
            continue

        key = normalized_key(get_primary_text_from_block(block))
        if not key:
            continue
        if len(key) > 120:
            continue
        if is_page_number_line(key):
            continue

        page_idx = block.get("page_idx")
        if isinstance(page_idx, int):
            text_to_pages.setdefault(key, set()).add(page_idx)

    min_repeated_pages = max(3, int(total_pages * 0.3))
    return {
        key for key, page_set in text_to_pages.items()
        if len(page_set) >= min_repeated_pages
    }


def clean_nested_value(value: Any, apply_corrections: bool = False) -> Any:
    if isinstance(value, str):
        cleaned = normalize_text(value, preserve_newlines=True)
        return correct_ocr_text(cleaned) if apply_corrections else cleaned

    if isinstance(value, list):
        cleaned: list[Any] = []
        for item in value:
            item = clean_nested_value(item, apply_corrections=apply_corrections)
            if item in ("", None, [], {}):
                continue
            cleaned.append(item)
        return cleaned

    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, item in value.items():
            item = clean_nested_value(item, apply_corrections=apply_corrections)
            if item in ("", None, [], {}):
                continue
            cleaned_dict[key] = item
        return cleaned_dict

    return value


def is_empty_after_clean(block: dict[str, Any]) -> bool:
    block_type = block.get("type")
    text = get_primary_text_from_block(block)

    if block_type in {"image", "chart", "table"}:
        return not bool(text or block_has_image_path(block))

    return not bool(text)


def find_output_file(tmp_root: Path, patterns: Iterable[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in tmp_root.rglob(pattern) if path.is_file())

    if not candidates:
        return None

    candidates.sort(key=lambda p: (len(str(p)), str(p)))
    return candidates[0]


def find_markdown_file(tmp_root: Path) -> Path | None:
    candidates = [
        path for path in tmp_root.rglob("*.md")
        if path.is_file() and not path.name.lower().startswith("readme")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(str(p)), str(p)))
    return candidates[0]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_paths_in_object(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        rewritten: dict[str, Any] = {}
        for key, value in obj.items():
            if key in IMAGE_PATH_FIELDS and isinstance(value, str):
                rewritten[key] = mapping.get(value, value)
            else:
                rewritten[key] = rewrite_paths_in_object(value, mapping)
        return rewritten

    if isinstance(obj, list):
        return [rewrite_paths_in_object(item, mapping) for item in obj]

    return obj


def collect_image_paths(obj: Any) -> set[str]:
    paths: set[str] = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in IMAGE_PATH_FIELDS and isinstance(value, str) and value.strip():
                paths.add(value.strip())
            else:
                paths.update(collect_image_paths(value))
    elif isinstance(obj, list):
        for item in obj:
            paths.update(collect_image_paths(item))

    return paths


def copy_assets_and_build_mapping(
    content_list: list[dict[str, Any]],
    content_list_path: Path,
    tmp_root: Path,
    assets_dir: Path,
    doc_id: str,
) -> dict[str, str]:
    original_paths = sorted(collect_image_paths(content_list))
    mapping: dict[str, str] = {}

    def find_source_image(original_path: str) -> Path | None:
        candidate = content_list_path.parent / original_path
        if candidate.exists() and candidate.is_file():
            return candidate

        basename = Path(original_path).name
        found = [p for p in tmp_root.rglob(basename) if p.is_file()]
        if found:
            found.sort(key=lambda p: (len(str(p)), str(p)))
            return found[0]

        return None

    for idx, original_path in enumerate(original_paths, start=1):
        source = find_source_image(original_path)
        if source is None:
            continue

        suffix = source.suffix or ".jpg"
        dest_name = f"{doc_id}__asset_{idx:06d}{suffix}"
        dest_path = assets_dir / dest_name

        if not dest_path.exists():
            shutil.copy2(source, dest_path)

        mapping[original_path] = f"assets/{dest_name}"

    return mapping


def clean_content_list(
    raw_content_list: list[dict[str, Any]],
    asset_mapping: dict[str, str],
    doc_id: str,
    source_pdf: str,
    source_filename: str,
    source_sha256: str,
    remove_repeated_margin: bool,
    apply_corrections: bool,
) -> list[dict[str, Any]]:
    repeated_margin_keys = (
        detect_repeated_margin_texts(raw_content_list)
        if remove_repeated_margin else set()
    )

    cleaned_blocks: list[dict[str, Any]] = []
    current_region = "body"

    for block_index, raw_block in enumerate(raw_content_list):
        if not isinstance(raw_block, dict):
            continue

        block_type = raw_block.get("type")
        if block_type in AUXILIARY_BLOCK_TYPES:
            continue

        block = rewrite_paths_in_object(raw_block, asset_mapping)
        block = clean_nested_value(block, apply_corrections=apply_corrections)

        raw_text_for_region = get_primary_text_from_block(block)
        if APPENDIX_RE.search(raw_text_for_region):
            current_region = "appendix_form"
        elif current_region == "body" and SUPPLEMENTARY_RE.search(raw_text_for_region):
            current_region = "supplementary"

        key = normalized_key(get_primary_text_from_block(block))
        if (
            remove_repeated_margin
            and key
            and key in repeated_margin_keys
            and block_is_margin_like(block)
        ):
            continue

        if is_empty_after_clean(block):
            continue

        block["doc_id"] = doc_id
        block["source_pdf"] = source_pdf
        block["source_filename"] = source_filename
        block["source_sha256"] = source_sha256
        block["parser"] = "mineru"
        block["cleaned"] = True
        block["original_block_index"] = block_index
        block["ocr_corrections_applied"] = apply_corrections
        block = enrich_block_quality_metadata(block, current_region=current_region)

        cleaned_blocks.append(block)

    return cleaned_blocks


def block_to_jsonl_record(block: dict[str, Any]) -> dict[str, Any]:
    page_idx = block.get("page_idx")
    text = get_primary_text_from_block(block)
    block_id = f"{block.get('doc_id')}::p{page_idx}::b{block.get('original_block_index')}"

    return {
        "doc_id": block.get("doc_id"),
        "source_pdf": block.get("source_pdf"),
        "source_filename": block.get("source_filename"),
        "source_sha256": block.get("source_sha256"),
        "parser": "mineru",
        "block_id": block_id,
        "page_idx": page_idx,
        "page_no": page_idx + 1 if isinstance(page_idx, int) else None,
        "block_type": block.get("type", "unknown"),
        "content_category": block.get("content_category"),
        "quality_flags": block.get("quality_flags", []),
        "ocr_suspicious": block.get("ocr_suspicious", False),
        "article_no": block.get("article_no"),
        "article_sub_no": block.get("article_sub_no"),
        "article_title": block.get("article_title"),
        "text": text,
        "bbox": block.get("bbox"),
        "asset_path": (
            block.get("img_path")
            or block.get("image_path")
            or block.get("table_img_path")
            or block.get("chart_img_path")
        ),
        "raw": block,
    }


def clean_markdown(md_text: str, asset_mapping: dict[str, str], apply_corrections: bool = False) -> str:
    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_RE.sub("", text)
    if apply_corrections:
        text = correct_ocr_text(text)

    lines: list[str] = []
    for line in text.split("\n"):
        line = MULTI_SPACE_RE.sub(" ", line).rstrip()
        if is_page_number_line(line):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = MULTI_BLANK_LINE_RE.sub("\n\n", text).strip() + "\n"

    for original_path, new_path in asset_mapping.items():
        # Markdown files are under parsed/markdown, so assets are one level up.
        md_relative_path = "../" + new_path
        text = text.replace(original_path, md_relative_path)

    return text


def output_exists(dirs: dict[str, Path], doc_id: str) -> bool:
    return (
        (dirs["json"] / f"{doc_id}.content_list.json").exists()
        and (dirs["markdown"] / f"{doc_id}.md").exists()
    )


def build_mineru_command(
    safe_pdf_path: Path,
    safe_output_dir: Path,
    backend: str,
    method: str,
    lang: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        "mineru",
        "-p", str(safe_pdf_path),
        "-o", str(safe_output_dir),
        "-m", method,
        "-b", backend,
        "-f", args.formula,
        "-t", args.table,
    ]

    if lang and lang.lower() != "none":
        command.extend(["-l", lang])

    if args.api_url:
        command.extend(["--api-url", args.api_url])

    return command


def make_attempts(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Return list of (attempt_name, backend, lang)."""
    lang = args.lang or "none"
    attempts: list[tuple[str, str, str]] = [("primary", args.backend, lang)]

    if not args.auto_fallback:
        return attempts

    seen = {(args.backend, lang)}

    def add(name: str, backend: str, lang_value: str) -> None:
        key = (backend, lang_value)
        if key not in seen:
            seen.add(key)
            attempts.append((name, backend, lang_value))

    # Safer backend fallback.
    if args.backend != "pipeline":
        add("fallback_pipeline", "pipeline", lang)

    # Language hint can sometimes cause OCR/model issues. Retry without it.
    if lang.lower() != "none":
        add("fallback_no_lang", args.backend, "none")
        if args.backend != "pipeline":
            add("fallback_pipeline_no_lang", "pipeline", "none")

    return attempts


def run_mineru_attempt(
    command: list[str],
    timeout_seconds: int,
    cuda: str | None = None,
) -> tuple[int, str, str, float]:
    timeout = None if timeout_seconds <= 0 else timeout_seconds
    start = time.perf_counter()

    env = os.environ.copy()
    # RunPod/GPU defaults. These make the child MinerU process see the same GPU
    # and avoid repeated Hugging Face transfer errors when hf_transfer is installed.
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
    )

    elapsed = time.perf_counter() - start
    return result.returncode, result.stdout or "", result.stderr or "", elapsed


def write_attempt_logs(
    logs_dir: Path,
    doc_id: str,
    attempt_index: int,
    attempt_name: str,
    command: list[str],
    stdout: str,
    stderr: str,
    returncode: int,
    elapsed_sec: float,
) -> None:
    prefix = f"{doc_id}.attempt{attempt_index:02d}.{attempt_name}"
    (logs_dir / f"{prefix}.stdout.log").write_text(stdout, encoding="utf-8")
    (logs_dir / f"{prefix}.stderr.log").write_text(stderr, encoding="utf-8")
    save_json(
        logs_dir / f"{prefix}.command.json",
        {
            "doc_id": doc_id,
            "attempt_index": attempt_index,
            "attempt_name": attempt_name,
            "command": command,
            "returncode": returncode,
            "elapsed_sec": round(elapsed_sec, 3),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def prepare_safe_work_paths(work_dir: Path, doc_id: str) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = work_dir / f"job_{doc_id}_{timestamp}"
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_pdf_path = input_dir / "input.pdf"
    return job_dir, safe_pdf_path, output_dir


def run_mineru_with_fallbacks(
    original_pdf_path: Path,
    doc_id: str,
    work_dir: Path,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[bool, Path | None, list[dict[str, Any]], Path]:
    job_dir, safe_pdf_path, safe_output_dir = prepare_safe_work_paths(work_dir, doc_id)

    # Critical filename/path insulation point.
    # MinerU receives only this ASCII-safe input.pdf, never the original filename.
    shutil.copy2(original_pdf_path, safe_pdf_path)

    attempts_log: list[dict[str, Any]] = []
    successful_output_dir: Path | None = None

    for attempt_index, (attempt_name, backend, lang) in enumerate(make_attempts(args), start=1):
        # Use a separate output directory per attempt to avoid mixed partial outputs.
        attempt_output_dir = safe_output_dir / f"attempt_{attempt_index:02d}"
        attempt_output_dir.mkdir(parents=True, exist_ok=True)

        command = build_mineru_command(
            safe_pdf_path=safe_pdf_path,
            safe_output_dir=attempt_output_dir,
            backend=backend,
            method=args.method,
            lang=lang,
            args=args,
        )

        try:
            returncode, stdout, stderr, elapsed = run_mineru_attempt(
                command=command,
                timeout_seconds=args.timeout,
                cuda=args.cuda,
            )
        except subprocess.TimeoutExpired as exc:
            returncode = -1
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr if isinstance(exc.stderr, str) else "") + "\n[TIMEOUT] MinerU attempt timed out."
            elapsed = float(args.timeout)

        write_attempt_logs(
            logs_dir=dirs["logs"],
            doc_id=doc_id,
            attempt_index=attempt_index,
            attempt_name=attempt_name,
            command=command,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            elapsed_sec=elapsed,
        )

        attempts_log.append(
            {
                "attempt_index": attempt_index,
                "attempt_name": attempt_name,
                "backend": backend,
                "lang": None if lang.lower() == "none" else lang,
                "returncode": returncode,
                "elapsed_sec": round(elapsed, 3),
                "stdout_log": str(dirs["logs"] / f"{doc_id}.attempt{attempt_index:02d}.{attempt_name}.stdout.log"),
                "stderr_log": str(dirs["logs"] / f"{doc_id}.attempt{attempt_index:02d}.{attempt_name}.stderr.log"),
            }
        )

        content_file = find_output_file(attempt_output_dir, ["*content_list*.json", "*_content_list.json"])
        if returncode == 0 and content_file is not None:
            successful_output_dir = attempt_output_dir
            return True, successful_output_dir, attempts_log, job_dir

    return False, None, attempts_log, job_dir


def finalize_outputs_from_mineru_tree(
    mineru_tree: Path,
    dirs: dict[str, Path],
    doc_id: str,
    source_pdf: str,
    source_filename: str,
    source_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    content_list_path = find_output_file(mineru_tree, ["*content_list*.json", "*_content_list.json"])
    middle_path = find_output_file(mineru_tree, ["*middle*.json", "*_middle.json"])
    markdown_path = find_markdown_file(mineru_tree)

    if content_list_path is None:
        raise ParserError("MinerU succeeded but content_list JSON was not found.")

    raw_content_list = load_json(content_list_path)
    if not isinstance(raw_content_list, list):
        raise ParserError("MinerU content_list JSON is not a list.")

    asset_mapping = copy_assets_and_build_mapping(
        content_list=raw_content_list,
        content_list_path=content_list_path,
        tmp_root=mineru_tree,
        assets_dir=dirs["assets"],
        doc_id=doc_id,
    )

    cleaned_content_list = clean_content_list(
        raw_content_list=raw_content_list,
        asset_mapping=asset_mapping,
        doc_id=doc_id,
        source_pdf=source_pdf,
        source_filename=source_filename,
        source_sha256=source_sha256,
        remove_repeated_margin=not args.disable_repeated_margin_clean,
        apply_corrections=args.enable_ocr_corrections,
    )

    final_content_path = dirs["json"] / f"{doc_id}.content_list.json"
    final_middle_path = dirs["json"] / f"{doc_id}.middle.json"
    final_markdown_path = dirs["markdown"] / f"{doc_id}.md"
    final_text_path = dirs["text"] / f"{doc_id}.pymupdf.txt"
    final_report_path = dirs["reports"] / f"{doc_id}.quality_report.json"

    save_json(final_content_path, cleaned_content_list)

    if middle_path is not None:
        middle_json = load_json(middle_path)
        middle_json = rewrite_paths_in_object(middle_json, asset_mapping)
        save_json(final_middle_path, middle_json)

    if markdown_path is not None:
        md_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
        final_markdown_path.write_text(
            clean_markdown(md_text, asset_mapping, apply_corrections=args.enable_ocr_corrections),
            encoding="utf-8",
        )
    else:
        final_markdown_path.write_text("", encoding="utf-8")

    pdf_text = ""
    pdf_text_info: dict[str, Any] = {"available": False, "reason": "secondary text disabled"}
    if args.secondary_text:
        pdf_text, pdf_text_info = extract_pdf_text_with_pymupdf(Path(source_pdf))
        if pdf_text:
            final_text_path.write_text(pdf_text, encoding="utf-8")

    if args.quality_report:
        quality_report = build_quality_report(
            cleaned_content_list=cleaned_content_list,
            pdf_text=pdf_text,
            pdf_text_info=pdf_text_info,
            doc_id=doc_id,
            source_pdf=source_pdf,
            source_filename=source_filename,
        )
        save_json(final_report_path, quality_report)

    if args.keep_raw:
        dirs["raw"].mkdir(parents=True, exist_ok=True)
        raw_dest = dirs["raw"] / doc_id
        if raw_dest.exists():
            shutil.rmtree(raw_dest)
        shutil.copytree(mineru_tree, raw_dest)

    return {
        "block_count": len(cleaned_content_list),
        "asset_count": len(asset_mapping),
        "markdown_path": str(final_markdown_path),
        "content_list_path": str(final_content_path),
        "middle_path": str(final_middle_path) if middle_path is not None else None,
        "secondary_text_path": str(final_text_path) if final_text_path.exists() else None,
        "quality_report_path": str(final_report_path) if final_report_path.exists() else None,
    }


def rebuild_all_blocks(json_dir: Path) -> int:
    all_blocks_path = json_dir / "all_blocks.jsonl"
    count = 0

    with all_blocks_path.open("w", encoding="utf-8") as out:
        for content_path in sorted(json_dir.glob("doc_*.content_list.json")):
            content_list = load_json(content_path)
            if not isinstance(content_list, list):
                continue

            for block in content_list:
                if not isinstance(block, dict):
                    continue
                out.write(json.dumps(block_to_jsonl_record(block), ensure_ascii=False) + "\n")
                count += 1

    return count


def process_pdf(pdf_path: Path, dirs: dict[str, Path], work_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    doc_id, sha = make_doc_id(pdf_path)
    started_at = datetime.now().isoformat(timespec="seconds")

    base_record: dict[str, Any] = {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path.resolve()),
        "source_filename": pdf_path.name,
        "source_sha256": sha,
        "parser": "mineru",
        "backend_requested": args.backend,
        "method": args.method,
        "lang_requested": None if args.lang.lower() == "none" else args.lang,
        "cuda_requested": args.cuda,
        "quality_report_enabled": args.quality_report,
        "secondary_text_enabled": args.secondary_text,
        "ocr_corrections_enabled": args.enable_ocr_corrections,
        "started_at": started_at,
    }

    if not args.force and output_exists(dirs, doc_id):
        return {
            **base_record,
            "status": "skipped",
            "reason": "already_parsed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "markdown_path": str(dirs["markdown"] / f"{doc_id}.md"),
            "content_list_path": str(dirs["json"] / f"{doc_id}.content_list.json"),
            "middle_path": str(dirs["json"] / f"{doc_id}.middle.json"),
        }

    ok, mineru_output_dir, attempts, job_dir = run_mineru_with_fallbacks(
        original_pdf_path=pdf_path,
        doc_id=doc_id,
        work_dir=work_dir,
        dirs=dirs,
        args=args,
    )

    try:
        if not ok or mineru_output_dir is None:
            return {
                **base_record,
                "status": "failed",
                "reason": "all_mineru_attempts_failed",
                "attempts": attempts,
                "work_dir": str(job_dir) if args.keep_work else None,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }

        output_info = finalize_outputs_from_mineru_tree(
            mineru_tree=mineru_output_dir,
            dirs=dirs,
            doc_id=doc_id,
            source_pdf=str(pdf_path.resolve()),
            source_filename=pdf_path.name,
            source_sha256=sha,
            args=args,
        )

        return {
            **base_record,
            "status": "success",
            "reason": None,
            "attempts": attempts,
            **output_info,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    except Exception as exc:
        return {
            **base_record,
            "status": "failed",
            "reason": f"finalize_failed: {type(exc).__name__}: {exc}",
            "attempts": attempts,
            "work_dir": str(job_dir) if args.keep_work else None,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    finally:
        if not args.keep_work:
            shutil.rmtree(job_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    try:
        check_mineru_cli()
        check_cuda_requirement(args)
    except ParserError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    dirs = ensure_output_dirs(output_dir)
    work_dir = select_work_dir(args.work_dir)
    pdfs = find_pdfs(
        input_dir,
        recursive=args.recursive,
        only=args.only,
        start_index=args.start_index,
        limit=args.limit,
    )

    print(f"[INFO] python    : {sys.executable}")
    print(f"[INFO] mineru    : {shutil.which('mineru')}")
    print(f"[INFO] input_dir : {input_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] work_dir  : {work_dir}")
    print(f"[INFO] pdf_count : {len(pdfs)}")
    print(f"[INFO] backend   : {args.backend}")
    print(f"[INFO] method    : {args.method}")
    print(f"[INFO] lang      : {args.lang}")
    print(f"[INFO] cuda      : {args.cuda}")
    print(f"[INFO] range     : start_index={args.start_index}, limit={args.limit}, only={args.only}")
    print(f"[INFO] quality   : report={args.quality_report}, secondary_text={args.secondary_text}, ocr_corrections={args.enable_ocr_corrections}")
    print("[INFO] MinerU input filenames are normalized to safe ASCII paths.")

    if not pdfs:
        print("[INFO] No PDFs found. Put PDFs in pdf_data and run again.")
        return 0

    manifest_path = dirs["manifests"] / "parse_manifest.jsonl"
    registry_path = dirs["manifests"] / "document_registry.jsonl"

    success = 0
    skipped = 0
    failed = 0

    for idx, pdf_path in enumerate(pdfs, start=1):
        doc_id, _ = make_doc_id(pdf_path)
        print(f"\n[{idx}/{len(pdfs)}] {doc_id}")
        print(f"  source: {pdf_path.name}")

        record = process_pdf(pdf_path, dirs, work_dir, args)
        append_jsonl(manifest_path, record)

        registry_record = {
            "doc_id": record.get("doc_id"),
            "source_pdf": record.get("source_pdf"),
            "source_filename": record.get("source_filename"),
            "source_sha256": record.get("source_sha256"),
            "status": record.get("status"),
            "content_list_path": record.get("content_list_path"),
            "markdown_path": record.get("markdown_path"),
            "secondary_text_path": record.get("secondary_text_path"),
            "quality_report_path": record.get("quality_report_path"),
            "created_at": record.get("finished_at"),
        }
        append_jsonl(registry_path, registry_record)

        status = record.get("status")
        if status == "success":
            success += 1
            print(f"  [OK] blocks={record.get('block_count')} assets={record.get('asset_count')}")
            if record.get("quality_report_path"):
                print(f"  report: {record.get('quality_report_path')}")
        elif status == "skipped":
            skipped += 1
            print("  [SKIP] already parsed")
        else:
            failed += 1
            print(f"  [FAIL] reason={record.get('reason')}")
            attempts = record.get("attempts") or []
            if attempts:
                last = attempts[-1]
                print(f"  last stderr: {last.get('stderr_log')}")

    total_blocks = rebuild_all_blocks(dirs["json"])

    print("\n[DONE]")
    print(f"  success   : {success}")
    print(f"  skipped   : {skipped}")
    print(f"  failed    : {failed}")
    print(f"  all_blocks: {dirs['json'] / 'all_blocks.jsonl'}")
    print(f"  block_cnt : {total_blocks}")
    print(f"  manifest  : {manifest_path}")
    print(f"  registry  : {registry_path}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
