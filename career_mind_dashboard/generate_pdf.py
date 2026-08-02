#!/usr/bin/env python3
"""Generate PDF report inside dashboard folder."""

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
PDF_OUTPUT = HERE / 'CareerMind_Report.pdf'


def pdf_exists() -> bool:
    """Check if PDF report has already been generated."""
    return PDF_OUTPUT.exists()


def get_pdf_path() -> str:
    """Return path to existing PDF or generate one if needed.
    
    Returns None if PDF generation fails (e.g., missing WeasyPrint dependencies).
    """
    if PDF_OUTPUT.exists():
        return str(PDF_OUTPUT)

    # Generate PDF using local dashboard generate_report.py
    try:
        from generate_report import main as generate_report_main
        generate_report_main()

        if PDF_OUTPUT.exists():
            return str(PDF_OUTPUT)
    except Exception as e:
        import warnings
        warnings.warn(f'PDF generation failed ({type(e).__name__}: {e}). PDF button will be unavailable.', stacklevel=2)
    
    return None


if __name__ == '__main__':
    pdf_path = get_pdf_path()
    if pdf_path:
        print(f'PDF ready: {pdf_path}')
    else:
        print('PDF generation failed')
        sys.exit(1)
