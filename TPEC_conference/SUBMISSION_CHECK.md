# TPEC 2027 submission-format check

Checked against https://tpec.engr.tamu.edu/submissions.html on October 9, 2026 (UTC). This records local technical checks, not conference acceptance or PDF eXpress certification.

| Requirement | Current manuscript |
|---|---|
| IEEE conference template | IEEEtran conference mode, 10-point text, two columns; no custom geometry |
| US Letter, 8.5 × 11 inches | Passed: all six pages are 612 × 792 PDF points |
| At most six pages, including figures/tables/references | Passed: six pages including references |
| All PDF fonts embedded; no Type 3 fonts | Passed: 23 embedded Type 1 fonts |
| Resolved references and citations | Passed after pdfLaTeX → BibTeX → two pdfLaTeX passes |
| Layout | All six rendered pages reviewed; no clipping or overlap found; final build has no overflow warnings |
| First-page copyright notice | Missing; add the applicable notice before camera-ready PDF validation |
| IEEE PDF eXpress Plus | Not performed; required by the conference before final proceedings submission |

The submission website currently lists `979-8-3315-5720-1/26/$31.00 ©2027 IEEE` for most authors, with separate government/Crown/EU categories. It asks for placement about 1 cm below the first-page left column, aligned with its left margin. The `/26` identifier and 2027 year are reproduced here exactly as posted; confirm the applicable final notice with the conference instructions rather than silently changing it. The page lists PDF eXpress conference ID `72677`.

The reviewed manuscript source was left unchanged during this check. Copyright transfer, PDF eXpress validation, and EasyChair submission remain author actions. Add any required notice to the LaTeX before producing the PDF for validation; the website says not to modify the PDF after validation or rename the verified file.
