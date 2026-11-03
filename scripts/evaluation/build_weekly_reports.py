"""Build the week 1-3 progress workbook from recorded metrics.

Run from the repository root:
    python -m scripts.evaluation.build_weekly_reports --out docs/reports/DriverGuard_Bao_cao_tuan_1_2_3.xlsx

Numbers in week 3 are read from the summaries in outputs/audits/<date>/ (written by
scripts.evaluation.summarize_gru_dann and the feature audit), so the workbook cannot drift
from the artifacts it cites. Weeks 1-2 keep the wording agreed in the 05/10/2026 version.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from scripts.evaluation.weekly_report_content import (
    MEMBERS,
    SUPERVISOR,
    WEEK1,
    WEEK2,
    week3_content,
)

NAVY, PALE, WHITE = "4C5F91", "E7EAF6", "FFFFFF"
WIDTHS = (7, 25, 50, 20, 20, 18, 47, 34)
COLUMNS = "ABCDEFGH"
THIN = Side(style="thin", color="C8CEDD")


def _band(sheet, row: int, title: str) -> None:
    sheet.merge_cells(f"A{row}:H{row}")
    cell = sheet[f"A{row}"]
    cell.value = title
    for column in COLUMNS:
        sheet[f"{column}{row}"].fill = PatternFill("solid", fgColor=NAVY)
    cell.font = Font(name="Arial", size=11, bold=True, color=WHITE)
    cell.alignment = Alignment(vertical="center")
    sheet.row_dimensions[row].height = 25


def _header(sheet, row: int, labels) -> None:
    for column, label in zip(COLUMNS, labels, strict=True):
        cell = sheet[f"{column}{row}"]
        cell.value = label or None
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(name="Arial", size=10, bold=True, color=WHITE)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[row].height = 30


def _row(sheet, row: int, values, height: int) -> None:
    fill = PatternFill("solid", fgColor=PALE if row % 2 else WHITE)
    for column, value in zip(COLUMNS, values, strict=True):
        cell = sheet[f"{column}{row}"]
        cell.value = value if value != "" else None
        cell.fill = fill
        cell.font = Font(name="Arial", size=10, color="111827")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    sheet.row_dimensions[row].height = height


def _paired_row(sheet, row: int, values, height: int) -> None:
    """Problem/plan tables use B:C, D:E and F:G as merged text columns."""
    _row(sheet, row, values, height)
    for pair in ("B{0}:C{0}", "D{0}:E{0}", "F{0}:G{0}"):
        sheet.merge_cells(pair.format(row))


def _text(sheet, row: int, value: str, size: int = 10, height: int = 30, color: str = "111827") -> None:
    sheet.merge_cells(f"A{row}:H{row}")
    cell = sheet[f"A{row}"]
    cell.value = value
    cell.font = Font(name="Arial", size=size, color=color)
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.row_dimensions[row].height = height


def week_sheet(book: Workbook, number: int, week: dict) -> None:
    sheet = book.create_sheet(f"Tuần {number}")
    sheet.sheet_view.showGridLines = False
    for column, width in zip(COLUMNS, WIDTHS, strict=True):
        sheet.column_dimensions[column].width = width
    sheet.merge_cells("A1:H1")
    sheet["A1"] = f"BÁO CÁO TIẾN ĐỘ TUẦN {number} - TIỂU LUẬN CNTT"
    sheet["A1"].font = Font(name="Arial", size=17, bold=True, color=NAVY)
    sheet.row_dimensions[1].height = 40
    _text(sheet, 2, week["subtitle"], size=11, height=24, color="334155")
    _band(sheet, 3, "THÔNG TIN CHUNG")
    info = (("Tên đề tài:", "DriverGuard"), ("Giảng viên HD:", SUPERVISOR),
            ("Sinh viên TH:", MEMBERS), ("Thời gian:", week["period"]))
    for offset, (label, value) in enumerate(info):
        sheet[f"A{4 + offset}"] = label
        sheet[f"C{4 + offset}"] = value
        sheet[f"A{4 + offset}"].font = Font(name="Arial", size=10, bold=True)
        sheet[f"C{4 + offset}"].font = Font(name="Arial", size=10)
    row = 9
    _band(sheet, row, f"MỤC TIÊU & KẾ HOẠCH TUẦN {number}")
    for goal in week["goals"]:
        row += 1
        _text(sheet, row, goal)
    row += 2
    _band(sheet, row, f"PHÂN CÔNG CHI TIẾT CÔNG VIỆC TUẦN {number}")
    row += 1
    _header(sheet, row, ("STT", "Hạng mục / Module", "Đầu việc chi tiết", "Người phụ trách",
                         "Dự kiến hoàn thành", "Trạng thái", "Kết quả / Deliverables", "Ghi chú"))
    for index, task in enumerate(week["tasks"], 1):
        row += 1
        _row(sheet, row, (index, *task), week.get("task_height", 70))
    row += 2
    _band(sheet, row, "KHÓ KHĂN, VƯỚNG MẮC & GIẢI PHÁP ĐỀ XUẤT")
    row += 1
    _header(sheet, row, ("STT", "Khó khăn gặp phải", "", "Hướng giải quyết", "",
                         "Xin ý kiến GVHD", "", "Ghi chú"))
    for index, (problem, solution, ask, note) in enumerate(week["problems"], 1):
        row += 1
        _paired_row(sheet, row, (index, problem, "", solution, "", ask, "", note), 70)
    row += 2
    _band(sheet, row, week["next_title"])
    row += 1
    _header(sheet, row, ("STT", "Hạng mục", "", "Đầu việc chuẩn bị", "", "Dự kiến", "", "Ghi chú"))
    for index, (item, work, when, note) in enumerate(week["next"], 1):
        row += 1
        _paired_row(sheet, row, (index, item, "", work, "", when, "", note), 65)
    row += 2
    _band(sheet, row, "CĂN CỨ VÀ GHI CHÚ")
    for source in week["sources"]:
        row += 1
        _text(sheet, row, source, size=9, height=35, color="475569")


def table_sheet(book: Workbook, title: str, headers, rows, widths) -> None:
    sheet = book.create_sheet(title)
    sheet.sheet_view.showGridLines = False
    columns = COLUMNS[:len(headers)]
    for column, width in zip(columns, widths, strict=True):
        sheet.column_dimensions[column].width = width
    for column, label in zip(columns, headers, strict=True):
        cell = sheet[f"{column}1"]
        cell.value = label
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(name="Arial", size=10, bold=True, color=WHITE)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for index, values in enumerate(rows, 2):
        fill = PatternFill("solid", fgColor=PALE if index % 2 else WHITE)
        for column, value in zip(columns, values, strict=True):
            cell = sheet[f"{column}{index}"]
            cell.value = value
            cell.fill = fill
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    sheet.freeze_panes = "A2"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pytest_counts(path: Path) -> dict:
    root = ElementTree.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    return {key: int(float(suite.get(key, 0))) for key in ("tests", "failures", "errors", "skipped")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/reports/DriverGuard_Bao_cao_tuan_1_2_3.xlsx"))
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/audits/20261009"))
    arguments = parser.parse_args()
    audit = arguments.audit_dir
    book = Workbook()
    book.remove(book.active)
    week_sheet(book, 1, WEEK1)
    week_sheet(book, 2, WEEK2)
    week3, explanation, evidence = week3_content(
        pilot=_json(audit / "gru_dann_pilot_summary.json"),
        full=_json(audit / "gru_dann_uta_full_window_summary.json"),
        first=_json(audit / "gru_dann_uta_full_v1_summary.json"),
        features=_json(audit / "uta_full_v002_feature_audit.json"),
        tests=pytest_counts(audit / "pytest.xml"))
    week_sheet(book, 3, week3)
    table_sheet(book, "Giải thích", ("Mục", "Giải thích"), explanation, (30, 120))
    table_sheet(book, "Minh chứng", ("Mã", "Nội dung", "File / vị trí"), evidence, (8, 60, 80))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    book.save(arguments.out)
    print(arguments.out)


if __name__ == "__main__":
    main()
