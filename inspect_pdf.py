from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pdfplumber  # pip install pdfplumber
except Exception:
    pdfplumber = None  # type: ignore

try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    PdfReader = None  # type: ignore


URL_RE = re.compile(r"(https?://[^\s)]+|www\.[^\s)]+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s)]+", re.IGNORECASE)


def _safe(s: Optional[str]) -> str:
    return s if s else ""


def normalize_for_entity_search(text: str) -> str:
    """
    Make URLs/DOIs/emails more discoverable by undoing common PDF line-break / spacing issues.
    """
    if not text:
        return ""

    t = text

    # Join line-broken URLs
    t = re.sub(r"(https?://\S+)\s*\n\s*(\S+)", r"\1\2", t, flags=re.IGNORECASE)
    t = re.sub(r"(www\.\S+)\s*\n\s*(\S+)", r"\1\2", t, flags=re.IGNORECASE)

    # Remove spaces around dots in domain-ish strings
    t = re.sub(r"www\s*\.\s*", "www.", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\.\s*(nl|com|org|edu|gov|net)\b", r".\1", t, flags=re.IGNORECASE)

    return t


def extract_annotation_urls(pdf_path: Path, max_pages: Optional[int] = None) -> List[str]:
    """
    Extract clickable link targets (URI actions) from PDF annotations.
    This often contains URLs/DOIs that are NOT present in the text layer.
    """
    if PdfReader is None:
        return []

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return []

    pages = reader.pages[:max_pages] if max_pages else reader.pages
    urls = set()

    for page in pages:
        annots = page.get("/Annots")
        if not annots:
            continue

        for annot in annots:
            try:
                obj = annot.get_object()
                action = obj.get("/A")
                if action and action.get("/S") == "/URI":
                    uri = action.get("/URI")
                    if uri:
                        urls.add(str(uri))
            except Exception:
                pass

    return sorted(urls)


def extract_with_pypdf(pdf_path: Path, max_pages: Optional[int] = None) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Install with: pip install pypdf")

    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages] if max_pages else reader.pages

    out: List[str] = []
    for i, p in enumerate(pages, start=1):
        try:
            out.append(_safe(p.extract_text()))
        except Exception as e:
            out.append(f"[pypdf] ERROR extracting page {i}: {e}")
    return out


def extract_with_pdfplumber(pdf_path: Path, max_pages: Optional[int] = None) -> List[str]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed. Install with: pip install pdfplumber")

    out: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for i, page in enumerate(pages, start=1):
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                out.append(text)
            except Exception as e:
                out.append(f"[pdfplumber] ERROR extracting page {i}: {e}")
    return out


def summarize_pages(pages: List[str]) -> List[Tuple[int, int, int]]:
    summary = []
    for idx, t in enumerate(pages, start=1):
        t = t or ""
        n_chars = len(t.strip())
        n_lines = 0 if not t.strip() else t.count("\n") + 1
        summary.append((idx, n_chars, n_lines))
    return summary


def find_entities(text: str):
    t = normalize_for_entity_search(text)
    urls = sorted(set(URL_RE.findall(t)))
    emails = sorted(set(EMAIL_RE.findall(t)))
    dois = sorted(set(DOI_RE.findall(t)))
    return urls, emails, dois


def print_page(text: str, page_num: int, show_lines: int = 80):
    lines = (text or "").splitlines()
    if not lines:
        print(f"\n--- Page {page_num} (empty) ---\n")
        return

    print(f"\n--- Page {page_num} (showing up to {show_lines} lines) ---\n")
    for i, line in enumerate(lines[:show_lines], start=1):
        print(f"{i:>3}: {line}")
    if len(lines) > show_lines:
        print(f"\n... ({len(lines) - show_lines} more lines not shown)\n")


def print_summary(summary: List[Tuple[int, int, int]], top_empty: int = 10):
    print("\nPage summary (chars/lines):\n")
    for page_num, n_chars, n_lines in summary:
        print(f"page {page_num:>3}: {n_chars:>6} chars | {n_lines:>4} lines")

    empties = [p for p in summary if p[1] < 50]
    if empties:
        print(f"\nPages with very little extracted text (<50 chars): {len(empties)}")
        print("First few:", ", ".join(str(p[0]) for p in empties[:top_empty]))


def resolve_pdf_path(user_input: str, data_dir: Path) -> Path:
    p = Path(user_input)

    if p.exists() and p.is_file():
        return p

    candidate = data_dir / user_input
    if candidate.exists() and candidate.is_file():
        return candidate

    if not user_input.lower().endswith(".pdf"):
        candidate2 = data_dir / f"{user_input}.pdf"
        if candidate2.exists() and candidate2.is_file():
            return candidate2

        p2 = Path(f"{user_input}.pdf")
        if p2.exists() and p2.is_file():
            return p2

    raise FileNotFoundError(f"PDF not found: {user_input} (tried direct path and {data_dir}/)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and inspect text from PDFs (papers) with pypdf or pdfplumber."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help="PDF filename (searched in --data-dir) OR a full/relative path to the PDF",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to search when you provide only a filename (default: data)",
    )
    parser.add_argument(
        "--engine",
        choices=["pypdf", "pdfplumber", "both"],
        default="both",
        help="Extraction engine to use",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages")
    parser.add_argument("--page", type=int, default=None, help="Print a specific page (1-indexed)")
    parser.add_argument("--lines", type=int, default=80, help="How many lines to show when printing a page")
    parser.add_argument(
        "--grep",
        type=str,
        default=None,
        help="Search for a regex string and show matching lines with page numbers",
    )
    parser.add_argument(
        "--entities",
        action="store_true",
        help="Print extracted URLs/emails/DOIs found in extracted text + annotation links",
    )
    parser.add_argument(
        "--annots",
        action="store_true",
        help="Print annotation URLs (clickable links) even if --entities is not set",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # If no pdf provided: list PDFs in data_dir
    if args.pdf is None:
        if not data_dir.exists():
            print(f"No PDF specified and data dir not found: {data_dir}", file=sys.stderr)
            sys.exit(1)

        pdfs = sorted(data_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF specified and no PDFs found in {data_dir}/.", file=sys.stderr)
            sys.exit(1)

        print(f"Pick a PDF (in {data_dir}/):")
        for p in pdfs:
            print(f"- {p.name}")
        sys.exit(0)

    try:
        pdf_path = resolve_pdf_path(args.pdf, data_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    def run_engine(name: str) -> List[str]:
        if name == "pypdf":
            return extract_with_pypdf(pdf_path, args.max_pages)
        if name == "pdfplumber":
            return extract_with_pdfplumber(pdf_path, args.max_pages)
        raise ValueError(name)

    def maybe_print_annotation_urls():
        ann_urls = extract_annotation_urls(pdf_path, args.max_pages)
        print("\nAnnotation URLs (clickable links):")
        if not ann_urls:
            print("  (none found)")
        else:
            for u in ann_urls:
                print(f"  - {u}")

    if args.engine in ("pypdf", "pdfplumber"):
        pages = run_engine(args.engine)
        summary = summarize_pages(pages)
        print(f"\nPDF: {pdf_path} | engine={args.engine} | pages_extracted={len(pages)}")
        print_summary(summary)

        full_text = "\n\n".join(pages)

        if args.entities:
            urls, emails, dois = find_entities(full_text)
            ann_urls = extract_annotation_urls(pdf_path, args.max_pages)

            # merge URLs from text + annotation
            all_urls = sorted(set(urls) | set(ann_urls))

            print("\nEntities found:")
            print(f"URLs in text ({len(urls)}):")
            for u in urls:
                print(f"  - {u}")

            print(f"\nURLs in annotations ({len(ann_urls)}):")
            for u in ann_urls:
                print(f"  - {u}")

            print(f"\nALL URLs merged ({len(all_urls)}):")
            for u in all_urls:
                print(f"  - {u}")

            print(f"\nEmails ({len(emails)}):")
            for e in emails:
                print(f"  - {e}")

            print(f"\nDOIs ({len(dois)}):")
            for d in dois:
                print(f"  - {d}")

        if args.annots and not args.entities:
            maybe_print_annotation_urls()

        if args.grep:
            pattern = re.compile(args.grep, re.IGNORECASE)
            print(f"\nMatches for /{args.grep}/i:\n")
            for page_idx, t in enumerate(pages, start=1):
                for line in (t or "").splitlines():
                    if pattern.search(line):
                        print(f"page {page_idx:>3}: {line}")

        if args.page:
            if 1 <= args.page <= len(pages):
                print_page(pages[args.page - 1], args.page, args.lines)
            else:
                print(f"--page out of range (1..{len(pages)})", file=sys.stderr)

        return

    # both
    pages_a = run_engine("pypdf")
    pages_b = run_engine("pdfplumber")
    n = min(len(pages_a), len(pages_b))

    print(f"\nPDF: {pdf_path} | comparing pypdf vs pdfplumber | pages={n}")
    sum_a = summarize_pages(pages_a[:n])
    sum_b = summarize_pages(pages_b[:n])

    print("\nComparison (chars):")
    for i in range(n):
        pa = sum_a[i][1]
        pb = sum_b[i][1]
        delta = pb - pa
        marker = "pdfplumber>>" if delta > 500 else ("pypdf>>" if delta < -500 else "≈")
        print(f"page {i+1:>3}: pypdf={pa:>6} | pdfplumber={pb:>6} | delta={delta:>6} | {marker}")

    if args.page:
        p = args.page
        if not (1 <= p <= n):
            print(f"--page out of range (1..{n})", file=sys.stderr)
            sys.exit(1)

        print_page(pages_a[p - 1], p, args.lines)
        print_page(pages_b[p - 1], p, args.lines)

    if args.entities:
        text_a = "\n\n".join(pages_a)
        text_b = "\n\n".join(pages_b)

        urls_a, emails_a, dois_a = find_entities(text_a)
        urls_b, emails_b, dois_b = find_entities(text_b)

        ann_urls = extract_annotation_urls(pdf_path, args.max_pages)

        print("\nEntities found (pypdf):")
        print(f"URLs ({len(urls_a)}):", ", ".join(urls_a[:20]) + (" ..." if len(urls_a) > 20 else ""))
        print(f"Emails ({len(emails_a)}):", ", ".join(emails_a[:20]) + (" ..." if len(emails_a) > 20 else ""))
        print(f"DOIs ({len(dois_a)}):", ", ".join(dois_a[:20]) + (" ..." if len(dois_a) > 20 else ""))

        print("\nEntities found (pdfplumber):")
        print(f"URLs ({len(urls_b)}):", ", ".join(urls_b[:20]) + (" ..." if len(urls_b) > 20 else ""))
        print(f"Emails ({len(emails_b)}):", ", ".join(emails_b[:20]) + (" ..." if len(emails_b) > 20 else ""))
        print(f"DOIs ({len(dois_b)}):", ", ".join(dois_b[:20]) + (" ..." if len(dois_b) > 20 else ""))

        print("\nAnnotation URLs (clickable links):")
        if not ann_urls:
            print("  (none found)")
        else:
            for u in ann_urls:
                print(f"  - {u}")

        merged = sorted(set(urls_a) | set(urls_b) | set(ann_urls))
        print(f"\nALL URLs merged (pypdf ∪ pdfplumber ∪ annots) ({len(merged)}):")
        for u in merged:
            print(f"  - {u}")

    if args.annots and not args.entities:
        # still show annots in "both" mode
        ann_urls = extract_annotation_urls(pdf_path, args.max_pages)
        print("\nAnnotation URLs (clickable links):")
        if not ann_urls:
            print("  (none found)")
        else:
            for u in ann_urls:
                print(f"  - {u}")

    if args.grep:
        pattern = re.compile(args.grep, re.IGNORECASE)
        print(f"\nMatches for /{args.grep}/i (pdfplumber):\n")
        for page_idx, t in enumerate(pages_b, start=1):
            for line in (t or "").splitlines():
                if pattern.search(line):
                    print(f"page {page_idx:>3}: {line}")


if __name__ == "__main__":
    main()
