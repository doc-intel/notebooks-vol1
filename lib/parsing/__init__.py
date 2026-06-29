"""Document parsing — Brick 1.

The package is split per format. Import from the format-specific subpackage:

    from lib.parsing.pdf import parse_pdf, fitz_pdf_to_line_df
    from lib.parsing.docx import parse_docx          # Volume 2
    from lib.parsing.xlsx import parse_xlsx          # Volume 2
    from lib.parsing.pptx import parse_pptx          # Volume 2
    from lib.parsing.mail import parse_mail          # Volume 2

Every format produces the same shape of output: a dictionary of DataFrames
(`line_df`, `page_df`, ...) plus a `parsing_summary` dict, so downstream bricks
(`question`, `retrieval`, `generation`) stay format-agnostic.
"""
