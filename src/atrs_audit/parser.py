"""Parse the structured HTML body returned by the GOV.UK Content API."""

from __future__ import annotations

import re
from html.parser import HTMLParser


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class SectionParser(HTMLParser):
    """Extract h3 fields and their following content from an ATRS record."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._heading_level: str | None = None
        self._heading_id = ""
        self._heading_parts: list[str] = []
        self._current_key: str | None = None
        self._current_parts: list[str] = []
        self.sections: dict[str, dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h2", "h3"}:
            self._flush_section()
            self._heading_level = tag
            self._heading_id = dict(attrs).get("id") or ""
            self._heading_parts = []
        elif tag in {"p", "li", "br", "div", "tr"} and self._current_key:
            self._current_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h2", "h3"} and self._heading_level == tag:
            heading = normalise_space("".join(self._heading_parts))
            if tag == "h3":
                key = self._heading_id or slugify(heading)
                # Duplicate ids occur in older standard versions.
                base = key
                suffix = 2
                while key in self.sections:
                    key = f"{base}-{suffix}"
                    suffix += 1
                self._current_key = key
                self.sections[key] = {"heading": heading, "value": ""}
            else:
                self._current_key = None
            self._heading_level = None
        elif tag in {"p", "li", "div", "tr"} and self._current_key:
            self._current_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._heading_level:
            self._heading_parts.append(data)
        elif self._current_key:
            self._current_parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush_section()

    def _flush_section(self) -> None:
        if self._current_key and self._current_parts:
            self.sections[self._current_key]["value"] = normalise_space(
                "".join(self._current_parts)
            )
        self._current_parts = []


def slugify(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def parse_sections(body_html: str) -> dict[str, dict[str, str]]:
    parser = SectionParser()
    parser.feed(body_html or "")
    parser.close()
    return parser.sections
