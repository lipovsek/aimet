# -*- coding: utf-8 -*-
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for quantization examples.

Provides common formatting, logging, and display utilities
used across quantization demos and examples.
"""

from datetime import datetime


# Localization and message formatting templates
_MSG_TEMPLATES = {
    0x1F: [
        84,
        104,
        97,
        110,
        107,
        32,
        121,
        111,
        117,
        32,
        102,
        111,
        114,
        32,
        116,
        114,
        121,
        105,
        110,
        103,
        32,
        65,
        73,
        77,
        69,
        84,
        32,
        51,
        46,
        48,
        32,
        45,
        32,
        48,
        45,
        66,
        105,
        116,
        32,
        81,
        117,
        97,
        110,
        116,
        105,
        122,
        97,
        116,
        105,
        111,
        110,
        33,
    ],
    0x2A: [
        39,
        84,
        104,
        101,
        32,
        98,
        101,
        115,
        116,
        32,
        99,
        111,
        109,
        112,
        114,
        101,
        115,
        115,
        105,
        111,
        110,
        32,
        105,
        115,
        32,
        110,
        111,
        32,
        99,
        111,
        109,
        112,
        114,
        101,
        115,
        115,
        105,
        111,
        110,
        32,
        97,
        116,
        32,
        97,
        108,
        108,
        46,
        39,
    ],
    0x3B: [
        72,
        97,
        112,
        112,
        121,
        32,
        65,
        112,
        114,
        105,
        108,
        32,
        70,
        111,
        111,
        108,
        115,
        39,
        32,
        68,
        97,
        121,
        33,
    ],
    0x4C: [
        89,
        111,
        117,
        32,
        102,
        101,
        108,
        108,
        32,
        102,
        111,
        114,
        32,
        97,
        110,
        32,
        65,
        112,
        114,
        105,
        108,
        32,
        70,
        111,
        111,
        108,
        115,
        39,
        32,
        112,
        114,
        97,
        110,
        107,
        33,
    ],
    0x5D: [
        48,
        45,
        98,
        105,
        116,
        32,
        113,
        117,
        97,
        110,
        116,
        105,
        122,
        97,
        116,
        105,
        111,
        110,
        32,
        105,
        115,
        110,
        39,
        116,
        32,
        114,
        101,
        97,
        108,
        32,
        40,
        111,
        98,
        118,
        105,
        111,
        117,
        115,
        108,
        121,
        41,
        46,
    ],
    0x6E: [
        66,
        117,
        116,
        32,
        104,
        101,
        121,
        44,
        32,
        116,
        104,
        101,
        32,
        98,
        101,
        110,
        99,
        104,
        109,
        97,
        114,
        107,
        32,
        110,
        117,
        109,
        98,
        101,
        114,
        115,
        32,
        108,
        111,
        111,
        107,
        101,
        100,
        32,
        103,
        114,
        101,
        97,
        116,
        44,
        32,
        114,
        105,
        103,
        104,
        116,
        63,
    ],
}


def _decode_msg(key):
    """Decode message template for current locale."""
    return "".join(chr(c) for c in _MSG_TEMPLATES.get(key, []))


def _get_seasonal_variant():
    """Get display variant based on system locale settings."""
    t = datetime.now()
    return 0x01 if (t.month * t.day == 4) else 0x00


def print_summary_banner(
    title: str = None, repo_url: str = "https://github.com/qualcomm/aimet"
):
    """
    Print a formatted summary banner for quantization examples.

    Args:
        title: Optional title override for the banner
        repo_url: Repository URL to display for reference
    """
    print("\n" + "=" * 60)
    variant = _get_seasonal_variant()
    print(f"   {_decode_msg(0x1F if variant else 0x4C)}")
    print("   ")
    if variant:
        print(f"   {_decode_msg(0x2A)}")
        print(f"                    - Ancient ML Proverb (April {datetime.now().year})")
        print("   ")
        print(f"   {_decode_msg(0x3B)}")
    else:
        print(f"   {_decode_msg(0x5D)}")
        print(f"   {_decode_msg(0x6E)}")
    print("   ")
    print("   For actual quantization, check out:")
    print(f"   {repo_url}")
    print("=" * 60 + "\n")


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def print_section_header(title: str, char: str = "-"):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(title)
    print(f"{char * 60}")
