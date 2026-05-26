from __future__ import annotations

from typing import Any, Dict


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_paper(record: Dict[str, Any], mode: str = "paper_raw_v1") -> str:
    title = _clean_text(record.get("title"))
    abstract = _clean_text(record.get("abstract"))
    category = _clean_text(record.get("category"))
    subcategory = _clean_text(record.get("subcategory"))
    date = _clean_text(record.get("date"))

    parts: list[str] = []

    if mode in {"paper_raw_v1", "paper_raw_v1_no_subcat", "paper_with_date_v1"}:
        parts.extend(["[Field]", category, ""])
        if mode != "paper_raw_v1_no_subcat":
            parts.extend(["[Subfield]", subcategory, ""])
        parts.extend(["[Title]", title, "", "[Abstract]", abstract])
        if mode == "paper_with_date_v1":
            parts.extend(["", "[PublicationDate]", date])
    elif mode == "title_only_v1":
        parts.extend(["[Title]", title])
    elif mode == "title_abstract_v1":
        parts.extend(["[Title]", title, "", "[Abstract]", abstract])
    else:
        raise ValueError(f"Unsupported formatter mode: {mode}")

    return "\n".join(parts).strip()


def format_idea_input(payload: Dict[str, Any], include_subfield: bool = True) -> str:
    field = _clean_text(payload.get("field"))
    subfield = _clean_text(payload.get("subfield"))
    title = _clean_text(payload.get("title"))
    description = _clean_text(payload.get("description"))

    parts: list[str] = ["[Field]", field, ""]
    if include_subfield and subfield:
        parts.extend(["[Subfield]", subfield, ""])
    parts.extend(["[Title]", title, "", "[Abstract]", description])
    return "\n".join(parts).strip()
