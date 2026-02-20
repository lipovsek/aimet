# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Report Writer for AIMET ONNX Regression Framework

This module generates comprehensive evaluation reports in both CSV and HTML formats.
It provides flexible column management, intelligent URL formatting, and interactive
HTML features for better data exploration.

Key Features:
- CSV generation with configurable column ordering and filtering
- Interactive HTML reports with client-side filtering and sorting
- Automatic job URL shortening for readability
- Safe HTML generation with proper escaping
- Support for hiding columns based on prefixes or specific keys

Design Philosophy:
Reports should be both machine-readable (CSV) and human-friendly (HTML).
The module prioritizes clarity and usability, automatically formatting
AI Hub job URLs as clickable links while maintaining clean visual presentation.

Technical Notes:
- Uses Jinja2 for safe HTML templating with auto-escaping
- Normalized key comparison for flexible column hiding
- Client-side JavaScript for responsive filtering without server round-trips
- Preserves original data while providing formatted views
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Mapping, Sequence, List, Optional, Dict, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape


# ==================== Utility Functions ====================


def _normalize_key(key: str) -> str:
    """
    Normalize dictionary keys for consistent comparison.

    This enables flexible column filtering by making keys case-insensitive
    and treating spaces/underscores as equivalent.

    Args:
        key: Original key string

    Returns:
        Normalized key (lowercase, spaces replaced with underscores)

    Example:
        >>> _normalize_key("AI Hub QNN Compile Job")
        "ai_hub_qnn_compile_job"
        >>> _normalize_key("qnn_latency")
        "qnn_latency"
    """
    return str(key).replace(" ", "_").lower()


def _extract_job_id(url: str) -> str:
    """
    Extract a concise job ID from an AI Hub URL.

    AI Hub URLs can be long and unwieldy in reports. This function
    extracts just the job ID portion for cleaner display while
    preserving the full URL as the link target.

    Args:
        url: Full AI Hub job URL

    Returns:
        Extracted job ID (last path segment) or truncated ID if very long

    Strategy:
        1. Parse URL and extract path segments
        2. Use the last non-empty path segment (typically the job ID)
        3. Remove query parameters and fragments
        4. Truncate to last 12 chars if ID is very long (>24 chars)

    Example:
        >>> _extract_job_id("https://hub.aihub.qualcomm.com/jobs/jg9jnwnq5/")
        "jg9jnwnq5"
    """
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)

        # Extract path segments
        segments = [s for s in parsed.path.split("/") if s]

        if segments:
            # Take the last segment as job ID
            candidate = segments[-1]
            # Remove any query or fragment parts
            candidate = candidate.split("?")[0].split("#")[0]

            # Truncate if very long (some jobs have UUID-style IDs)
            if len(candidate) > 24:
                return candidate[-12:]

            return candidate if candidate else url

        return url

    except Exception:
        # If parsing fails, return original URL
        return url


def as_clickable_job_id(url: str) -> Markup:
    """
    Convert an AI Hub URL to a clickable link showing just the job ID.

    This creates user-friendly HTML where the displayed text is just
    the job ID, but clicking it opens the full AI Hub URL.

    Args:
        url: Full URL to convert

    Returns:
        Safe HTML markup with clickable job ID

    Safety:
        - URLs are properly escaped to prevent XSS
        - Non-HTTP URLs are displayed as plain text
        - Uses target="_blank" with rel="noopener" for security
    """
    # Only process HTTP(S) URLs
    if not isinstance(url, str) or not url.startswith("http"):
        return Markup(escape(str(url)))

    # Extract job ID for display
    job_id = _extract_job_id(url)

    # Create safe HTML with escaped URL
    return Markup(
        f'<a href="{escape(url)}" target="_blank" rel="noopener">{escape(job_id)}</a>'
    )


def linkify(text: Any) -> Markup:
    """
    Convert plain URLs in text to clickable links.

    This function intelligently handles various input types:
    - Already marked-up HTML (Markup objects) are preserved
    - Plain text URLs are converted to links
    - Non-string values are converted to strings first

    Args:
        text: Input text that may contain URLs

    Returns:
        Safe HTML with URLs converted to clickable links

    Technical Note:
        Uses regex to find URLs but ensures all text is properly
        escaped to prevent XSS attacks.
    """
    # Already marked up - return as-is
    if isinstance(text, Markup):
        return text

    # Convert to string if needed
    if not isinstance(text, str):
        text = str(text)

    # Regex for finding URLs
    url_pattern = re.compile(r"(https?://\S+)")

    def replace_url(match):
        """Replace URL with clickable link."""
        url = match.group(1)
        return Markup(
            f'<a href="{url}" target="_blank" rel="noopener">{escape(url)}</a>'
        )

    # First escape the entire text, then replace URLs
    # This ensures non-URL text is safe
    escaped_text = escape(text)
    return Markup(url_pattern.sub(replace_url, escaped_text))


# ==================== CSV Generation ====================


def write_csv(
    results: Sequence[Mapping],
    csv_path: str,
    *,
    key_order: Optional[List[str]] = None,
    hide_prefixes: Optional[List[str]] = None,
    hide_keys: Optional[List[str]] = None,
) -> None:
    """
    Write evaluation results to a CSV file with flexible column management.

    This function generates clean CSV reports with configurable column
    visibility and ordering. It's designed to handle the varying output
    formats from different evaluation scenarios.

    Args:
        results: Sequence of result dictionaries (one per evaluation)
        csv_path: Output path for CSV file
        key_order: Optional list specifying column order (others appended)
        hide_prefixes: List of key prefixes to hide (e.g., ["qnn_"] hides all QNN columns)
        hide_keys: List of specific keys to hide

    Column Filtering:
        Keys are normalized (lowercase, spaces→underscores) before comparison.
        This allows hiding "AI Hub QNN Compile Job" by specifying "ai_hub_qnn".

    Example:
        >>> results = [
        ...     {"Model": "resnet50", "Accuracy": 0.92, "qnn_latency": 45.2},
        ...     {"Model": "mobilenet", "Accuracy": 0.78, "qnn_latency": 12.3}
        ... ]
        >>> write_csv(
        ...     results,
        ...     "report.csv",
        ...     key_order=["Model", "Accuracy"],
        ...     hide_prefixes=["qnn_"]  # Hide QNN columns for host-only report
        ... )

    File Safety:
        - Parent directories are created if they don't exist
        - UTF-8 encoding for international character support
        - Proper CSV escaping for special characters
    """
    # Normalize hiding criteria
    hide_prefixes = [_normalize_key(p) for p in (hide_prefixes or [])]
    hide_keys_set = set(_normalize_key(k) for k in (hide_keys or []))

    def should_keep_key(key: str) -> bool:
        """Determine if a column should be included."""
        normalized = _normalize_key(key)

        # Check explicit hide list
        if normalized in hide_keys_set:
            return False

        # Check prefix filters
        for prefix in hide_prefixes:
            if normalized.startswith(prefix):
                return False

        return True

    # Process results and track column order
    filtered_rows: List[Mapping] = []
    seen_keys: List[str] = []

    for row in results:
        filtered_row = {}
        for key, value in row.items():
            if not should_keep_key(key):
                continue

            filtered_row[key] = value

            # Track key order as they appear
            if key not in seen_keys:
                seen_keys.append(key)

        filtered_rows.append(filtered_row)

    # Determine final column order
    if key_order:
        # Start with specified order (if keys exist)
        header = [k for k in key_order if k in seen_keys]
        # Append remaining keys in order seen
        for key in seen_keys:
            if key not in header:
                header.append(key)
    else:
        # Use natural order (as keys appeared)
        header = seen_keys

    # Create output directory if needed
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Write CSV with determined column order
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row in filtered_rows:
            # Write row with proper column order, empty string for missing keys
            writer.writerow({k: row.get(k, "") for k in header})


# ==================== HTML Generation ====================


def write_html(
    results: Sequence[Mapping],
    html_path: str,
    *,
    key_order: Optional[List[str]] = None,
    hide_prefixes: Optional[List[str]] = None,
    hide_keys: Optional[List[str]] = None,
    page_title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    """
    Generate an interactive HTML report with filtering and sorting capabilities.

    Creates a professional-looking HTML report with:
    - Client-side filtering by model and feature
    - Sortable columns (model and feature)
    - Automatic job URL formatting as clickable IDs
    - Responsive design with hover effects
    - Clean, modern styling

    Args:
        results: Sequence of result dictionaries
        html_path: Output path for HTML file
        key_order: Optional column ordering
        hide_prefixes: Prefixes of columns to hide
        hide_keys: Specific columns to hide
        page_title: HTML page title (default: "AIMET PTQ Report")
        subtitle: Optional subtitle line (e.g., timestamp, device info)

    Special Handling:
        - Keys containing "job" with HTTP URLs are auto-formatted as clickable job IDs
        - Numeric values get special formatting for alignment
        - All text is properly escaped for XSS prevention

    Template System:
        Uses Jinja2 with auto-escaping for security. The template is expected
        at AIMETRegression/report/templates/report_template.html

    Example:
        >>> write_html(
        ...     results,
        ...     "report.html",
        ...     page_title="ResNet50 Quantization Results",
        ...     subtitle=f"Device: Samsung Galaxy S24 | {datetime.now()}"
        ... )
    """
    # Normalize hiding criteria
    hide_prefixes = [_normalize_key(p) for p in (hide_prefixes or [])]
    hide_keys_set = set(_normalize_key(k) for k in (hide_keys or []))

    def should_keep_key(key: str) -> bool:
        """Check if column should be displayed."""
        normalized = _normalize_key(key)

        if normalized in hide_keys_set:
            return False

        for prefix in hide_prefixes:
            if normalized.startswith(prefix):
                return False

        return True

    # Process results with special formatting
    processed_rows: List[Mapping] = []
    key_seen_order: List[str] = []

    for row in results:
        new_row = {}

        for key, value in row.items():
            if not should_keep_key(key):
                continue

            normalized_key = _normalize_key(key)

            # Special handling for job URLs
            # Auto-detect AI Hub job URLs and format them
            if (
                isinstance(value, str)
                and value.startswith("http")
                and "job" in normalized_key
            ):
                value = as_clickable_job_id(value)

            new_row[key] = value

            # Track column order
            if key not in key_seen_order:
                key_seen_order.append(key)

        processed_rows.append(new_row)

    # Determine column order
    if key_order:
        ordered_keys = [k for k in key_order if k in key_seen_order]
        # Append any additional keys not in specified order
        for key in key_seen_order:
            if key not in ordered_keys:
                ordered_keys.append(key)
    else:
        ordered_keys = key_seen_order

    # Set up Jinja2 template environment
    template_dir = "AIMETRegression/report/templates"
    template_name = "report_template.html"

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "htm"]),  # Auto-escape for security
    )

    # Register custom filters
    env.filters["linkify"] = linkify

    # Load and render template
    template = env.get_template(template_name)
    rendered = template.render(
        results=processed_rows,
        ordered_keys=ordered_keys,
        page_title=page_title,
        subtitle=subtitle,
    )

    # Write rendered HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(rendered)
