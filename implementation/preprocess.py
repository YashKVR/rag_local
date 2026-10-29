"""Turn the company PDFs in knowledge-files/ into retrieval chunks.

The PDF is the source. Each page's text layer is read with PyMuPDF, then split
on the document's own headings or table rows. The wording in the file is kept,
so dates, amounts, and grades are not rewritten by a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


KNOWLEDGE_FILES = Path(__file__).parent.parent / "knowledge-files"
MAX_CHARS = 1800
OVERLAP = 250

HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+?)\s*$")
DATE_RE = re.compile(r"\d{2}-\d{2}-\d{4}")
STOPWORDS = {"of", "and", "the", "for", "to", "a", "an", "on", "in", "per", "or"}
CITIES = ("MUMBAI", "NASHIK", "DELHI", "KOLKATA", "CHENNAI", "HYDERABAD")


@dataclass
class Chunk:
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Line:
    page: int
    text: str
    y: float


def build_chunks(folder: Path = KNOWLEDGE_FILES) -> list[Chunk]:
    paths = sorted(folder.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {folder}")

    chunks: list[Chunk] = []
    for path in paths:
        file_chunks = chunk_pdf(path)
        print(f"{path.name}: {len(file_chunks)} chunks", flush=True)
        chunks.extend(file_chunks)
    print(f"Prepared {len(chunks)} chunks from {len(paths)} PDFs", flush=True)
    return chunks


def chunk_pdf(path: Path) -> list[Chunk]:
    document = pymupdf.open(path)
    try:
        kind = classify(path)
        if kind == "holiday":
            chunks = chunk_holiday(path, document)
        elif kind == "promotion":
            chunks = chunk_promotion(path, document)
        elif kind == "page":
            chunks = chunk_by_page(path, document)
        else:
            chunks = chunk_policy(path, document)
        if chunks:
            return chunks
        print(f"{path.name}: structured split was empty, using one chunk per page", flush=True)
        return chunk_by_page(path, document)
    finally:
        document.close()


def classify(path: Path) -> str:
    name = path.name.lower()
    if "holiday" in name:
        return "holiday"
    if "promotion" in name:
        return "promotion"
    if "corporate" in name or "quick guide" in name or "insurance" in name:
        return "page"
    return "policy"


def chunk_policy(path: Path, document) -> list[Chunk]:
    lines = reading_lines(document)
    lines = [line for line in lines if not is_boilerplate(line.text)]
    if not lines:
        return []

    title = document_title(path)
    sections: list[tuple[str, int, list[str]]] = []
    current_name = "Overview"
    current_page = lines[0].page
    current: list[str] = []

    for line in lines:
        if is_heading(line.text):
            if current:
                sections.append((current_name, current_page, current))
            current_name = line.text.strip().rstrip(":")
            current_page = line.page
            current = [line.text.strip()]
        else:
            current.append(line.text)

    if current:
        sections.append((current_name, current_page, current))

    chunks: list[Chunk] = []
    for name, page, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if len(body) < 40:
            continue
        parts = split_long(body)
        for index, part in enumerate(parts):
            section = name if len(parts) == 1 else f"{name} ({index + 1}/{len(parts)})"
            chunks.append(make_chunk(title, section, part, path, page, "policy"))
    return chunks


def chunk_by_page(path: Path, document) -> list[Chunk]:
    title = document_title(path)
    chunks: list[Chunk] = []
    for number, page in enumerate(document, start=1):
        text = page_text(page)
        if len(text) < 30:
            continue
        chunks.append(make_chunk(title, f"page {number}", text, path, number, "page"))
    return chunks


def chunk_holiday(path: Path, document) -> list[Chunk]:
    title = document_title(path)
    chunks: list[Chunk] = []
    kind = "compulsory"
    for number, page in enumerate(document, start=1):
        words = page.get_text("words")
        header = [word for word in words if word_text(word).upper() in {*CITIES, "DATE", "DAY", "PARTICULARS"}]
        if not any(word_text(word).upper() == "MUMBAI" for word in header):
            continue
        city_x = {word_text(word).upper(): word[0] for word in header if word_text(word).upper() in CITIES}
        date_x = next(word[0] for word in header if word_text(word).upper() == "DATE")
        for line in cluster_lines(words):
            upper = line["text"].upper()
            if "OPTIONAL" in upper and not DATE_RE.search(line["text"]):
                kind = "optional"
                continue
            if "COMPULSORY" in upper and not DATE_RE.search(line["text"]):
                kind = "compulsory"
                continue
            date = next((word["text"] for word in line["words"] if DATE_RE.fullmatch(word["text"])), "")
            if not date:
                continue
            particulars = " ".join(
                word["text"] for word in line["words"] if 80 <= word["x"] < date_x - 15 and not word["text"].isdigit()
            ).strip(" -/")
            day = next(
                (word["text"] for word in line["words"] if word["text"].isalpha() and word["x"] > date_x + 30),
                "",
            )
            places = []
            for word in line["words"]:
                if "_" not in word["text"] and word["text"] not in {"_/", "/"}:
                    continue
                city = min(city_x, key=lambda name: abs(city_x[name] - word["x"]))
                if abs(city_x[city] - word["x"]) <= 45 and city.title() not in places:
                    places.append(city.title())
            if not particulars:
                continue
            where = ", ".join(places) if places else "no listed office"
            article = "an" if kind == "optional" else "a"
            body = (
                f"{particulars.title()} is {article} {kind} holiday on {day.title()} {date}. "
                f"Offices observing it: {where}."
            )
            chunks.append(make_chunk(title, particulars.title(), body, path, number, "holiday"))
    return chunks


def chunk_promotion(path: Path, document) -> list[Chunk]:
    title = document_title(path)
    chunks: list[Chunk] = []
    columns = None
    for number, page in enumerate(document, start=1):
        words = page.get_text("words")
        detected = promotion_columns(words)
        if detected is not None:
            columns = detected
            header_bottom = detected["header_bottom"]
        elif columns is None:
            continue
        else:
            # Later pages continue the same table without repeating the header.
            header_bottom = 0
        anchors = []
        for line in cluster_lines(words):
            if line["y"] <= header_bottom:
                continue
            serial = next((word for word in line["words"] if word["x"] < columns["id"] and word["text"].isdigit()), None)
            employee_id = next(
                (
                    word
                    for word in line["words"]
                    if columns["id"] <= word["x"] < columns["name"] and re.fullmatch(r"\d{3,6}", word["text"])
                ),
                None,
            )
            if serial and employee_id:
                anchors.append({"y": line["y"], "serial": serial["text"]})
        if not anchors:
            continue

        grouped: dict[float, list] = {anchor["y"]: [] for anchor in anchors}
        ys = [anchor["y"] for anchor in anchors]
        for word in words:
            y = word[1]
            if y <= header_bottom:
                continue
            nearest = min(ys, key=lambda anchor_y: abs(anchor_y - y))
            grouped[nearest].append({"x": word[0], "y": y, "text": word_text(word)})

        for anchor in anchors:
            cells = {name: [] for name in ("id", "name", "old_grade", "new_grade", "designation", "location")}
            for word in grouped[anchor["y"]]:
                column = column_for(word["x"], columns)
                if column in cells:
                    cells[column].append(word)
            employee_id = join_cell(cells["id"])
            name = join_cell(cells["name"])
            old_grade = join_cell(cells["old_grade"])
            new_grade = join_cell(cells["new_grade"])
            designation = join_cell(cells["designation"])
            location = join_cell(cells["location"])
            if not name:
                continue
            body = (
                f"{name} (employee id {employee_id}) was promoted from grade {old_grade} to grade {new_grade}. "
                f"New designation: {designation}. Location: {location}."
            )
            chunks.append(make_chunk(title, name, body, path, number, "promotion"))
    return chunks


def promotion_columns(words) -> dict | None:
    """Column x positions from the header row.

    Later body rows also contain words like "Sr.", so header tokens count only
    when they sit on the same band as the DESIGNATION label.
    """
    designation = [
        word for word in words if word_text(word).upper().startswith("DESIGNATION") and word[1] < 150
    ]
    if not designation:
        return None
    header_y = min(word[1] for word in designation)
    labels = {}
    grades = []
    news = []
    label_bottoms = []
    for word in words:
        if abs(word[1] - header_y) > 16:
            continue
        text = word_text(word).upper().strip(".")
        label_bottoms.append(word[3])
        if text == "GRADE":
            grades.append(word[0])
        elif text == "NEW":
            news.append(word[0])
        elif text in {"DESIGNATION", "LOCATION", "NAME", "ID", "EMP"}:
            labels[text] = word[0]
    if "DESIGNATION" not in labels or "LOCATION" not in labels or len(grades) < 2:
        return None
    designation_x = min((x for x in news if abs(x - labels["DESIGNATION"]) < 50), default=labels["DESIGNATION"])
    return {
        "header_bottom": max(label_bottoms) + 4,
        "id": labels.get("ID", labels.get("EMP", 70)) - 12,
        "name": labels.get("NAME", 140) - 20,
        "old_grade": min(grades) - 12,
        "new_grade": max(grades) - 12,
        # "Sr." in a designation starts left of the DESIGNATION label.
        "designation": designation_x - 26,
        "location": labels["LOCATION"] - 25,
    }


def column_for(x: float, columns: dict) -> str:
    order = ("location", "designation", "new_grade", "old_grade", "name", "id")
    for name in order:
        if x >= columns[name]:
            return name
    return "sr"


def join_cell(words: list[dict]) -> str:
    ordered = sorted(words, key=lambda word: (word["y"], word["x"]))
    return " ".join(word["text"] for word in ordered).strip()


def reading_lines(document) -> list[Line]:
    lines: list[Line] = []
    for number, page in enumerate(document, start=1):
        clustered = cluster_lines(page.get_text("words"))
        previous_y = None
        for line in clustered:
            if previous_y is not None and line["y"] - previous_y > 18:
                lines.append(Line(number, "", line["y"]))
            previous_y = line["y"]
            text = clean_text(line["text"])
            if text:
                lines.append(Line(number, text, line["y"]))
    return lines


def page_text(page) -> str:
    lines = []
    previous_y = None
    for line in cluster_lines(page.get_text("words")):
        if previous_y is not None and line["y"] - previous_y > 18:
            lines.append("")
        previous_y = line["y"]
        text = clean_text(line["text"])
        if text and not is_boilerplate(text):
            lines.append(text)
    return "\n".join(lines).strip()


def cluster_lines(words, y_tolerance: float = 3.5) -> list[dict]:
    lines: list[dict] = []
    for word in sorted(words, key=lambda item: (item[1], item[0])):
        text = word_text(word).strip()
        if not text:
            continue
        y = word[1]
        if not lines or abs(y - lines[-1]["y"]) > y_tolerance:
            lines.append({"y": y, "words": []})
        lines[-1]["words"].append({"x": word[0], "y": y, "text": text})
    for line in lines:
        line["words"].sort(key=lambda item: item["x"])
        line["text"] = " ".join(item["text"] for item in line["words"])
    return lines


def is_heading(text: str) -> bool:
    match = HEADING_RE.match(text.strip())
    if not match:
        return False
    title = match.group(2).strip(" :")
    if len(title) > 90 or len(title.split()) > 14:
        return False
    words = re.findall(r"[A-Za-z']+", title)
    ordinary = [word for word in words if word.islower() and word.lower() not in STOPWORDS]
    return len(ordinary) < 3


def is_boilerplate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"\d{1,3}", stripped):
        return True
    lowered = stripped.lower()
    if lowered in {"hirect limited", "leave policy", "gratuity policy"}:
        return True
    if "hirect" in lowered and "|" in stripped and len(stripped) < 80:
        return True
    return False


def split_long(text: str, limit: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            break_at = text.rfind("\n", start + limit // 2, end)
            if break_at == -1:
                break_at = text.rfind(" ", start + limit // 2, end)
            if break_at != -1:
                end = break_at
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return parts


def make_chunk(title: str, section: str, body: str, path: Path, page: int, kind: str) -> Chunk:
    body = body.strip()
    summary = re.sub(r"\s+", " ", body)[:280]
    headline = f"{title} — {section}"
    return Chunk(
        page_content=f"{headline}\n\n{summary}\n\n{body}",
        metadata={
            "source": path.name,
            "path": path.as_posix(),
            "page": int(page),
            "section": section,
            "type": kind,
        },
    )


def document_title(path: Path) -> str:
    stem = re.sub(r"\d{6,}", " ", path.stem)
    stem = stem.replace("_compressed", " ").replace("-", " ").replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" -")
    return stem


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\f", " ")
    return re.sub(r"\s+", " ", text).strip()


def word_text(word) -> str:
    return word[4]
