"""
Document chunker and metadata extractor for FinGuard AI Policy Knowledge Base.
Splits regulatory frameworks and SOPs into semantic chunks with metadata tracking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


def chunk_policy_document(
    file_path: Path | str, target_chunk_size: int = 350, chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Parse markdown policy file into semantic chunks preserving section headers.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    content = path.read_text(encoding="utf-8")
    doc_name = path.stem.replace("_", " ").title()

    # Split by markdown headers (### or ##)
    sections = re.split(r"\n(?=#{2,3}\s+)", content)
    chunks = []
    chunk_counter = 1

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        lines = sec.split("\n")
        header_match = re.match(r"^#{2,3}\s+(.+)$", lines[0])
        section_title = header_match.group(1).strip() if header_match else "General Provisions"
        body = "\n".join(lines[1:]).strip() if header_match else sec

        words = body.split()
        if len(words) <= target_chunk_size:
            chunk_id = f"{path.stem}_chk_{chunk_counter:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "source_doc": doc_name,
                "source_file": path.name,
                "section_title": section_title,
                "text": f"{section_title}\n{body}",
                "word_count": len(words),
            })
            chunk_counter += 1
        else:
            # Sliding window over large sections
            start = 0
            while start < len(words):
                end = min(start + target_chunk_size, len(words))
                sub_text = " ".join(words[start:end])
                chunk_id = f"{path.stem}_chk_{chunk_counter:03d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_doc": doc_name,
                    "source_file": path.name,
                    "section_title": section_title,
                    "text": f"{section_title} (Part {chunk_counter})\n{sub_text}",
                    "word_count": len(sub_text.split()),
                })
                chunk_counter += 1
                if end == len(words):
                    break
                start += target_chunk_size - chunk_overlap

    return chunks


def load_all_policy_chunks(policies_dir: Path | str = "data/policies") -> List[Dict[str, Any]]:
    """
    Load and chunk all markdown documents in the policies directory.
    """
    p_dir = Path(policies_dir)
    all_chunks = []
    for md_file in sorted(p_dir.glob("*.md")):
        chunks = chunk_policy_document(md_file)
        all_chunks.extend(chunks)
    return all_chunks
