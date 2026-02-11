"""Document generation node — creates PDF, Excel, or HTML from workflow data.

Output: StagedFile reference that can be downloaded or sent via channels.
"""

import io
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from ..base_node import BaseNode
from ..models import NodeConfig, NodeExecutionResult, NodeStatus
from events.event_bus import event_bus

logger = logging.getLogger(__name__)

STAGED_DIR = "/tmp/aictrlnet/staged_files"


class DocGenerationNode(BaseNode):
    """Node for generating documents from workflow data.

    Generates PDF (reportlab), Excel (openpyxl), or HTML documents from
    structured data produced by upstream nodes.

    Output includes a staged file reference that can be used for
    download or sent via messaging channels.
    """

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document generation."""
        start_time = datetime.utcnow()

        try:
            output_format = self.config.parameters.get("format", "pdf").lower()
            title = self.config.parameters.get("title", "Generated Document")
            filename = self.config.parameters.get("filename") or f"{title.replace(' ', '_')}.{output_format}"

            # Get data to render — from input or config
            data = input_data.get("extracted", input_data.get("data", input_data))
            template = self.config.parameters.get("template")

            if output_format == "pdf":
                file_bytes, content_type = self._generate_pdf(title, data, template)
            elif output_format in ("xlsx", "excel"):
                file_bytes, content_type = self._generate_excel(title, data)
                filename = filename.replace(".pdf", ".xlsx") if filename.endswith(".pdf") else filename
            elif output_format == "html":
                file_bytes, content_type = self._generate_html(title, data, template)
                filename = filename.replace(".pdf", ".html") if filename.endswith(".pdf") else filename
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            # Stage the generated file
            file_id = uuid.uuid4()
            os.makedirs(STAGED_DIR, exist_ok=True)
            storage_path = os.path.join(STAGED_DIR, str(file_id))
            with open(storage_path, "wb") as f:
                f.write(file_bytes)

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            await event_bus.publish(
                "node.executed",
                {
                    "node_id": self.config.id,
                    "node_type": "docGeneration",
                    "format": output_format,
                    "file_size": len(file_bytes),
                    "duration_ms": duration_ms,
                },
            )

            return {
                "file_id": str(file_id),
                "filename": filename,
                "content_type": content_type,
                "file_size": len(file_bytes),
                "storage_path": storage_path,
                "format": output_format,
            }

        except Exception as e:
            logger.error(f"DocGenerationNode {self.config.id} failed: {e}")
            raise

    def _generate_pdf(self, title: str, data: Any, template: Any = None) -> tuple:
        """Generate PDF using reportlab."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
        except ImportError:
            raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, title=title)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph(title, styles["Title"]))
        elements.append(Spacer(1, 12))

        # Render data
        if isinstance(data, dict):
            if "rows" in data and "headers" in data:
                # Tabular data
                elements.extend(self._pdf_table(data["headers"], data["rows"], styles))
            elif "sheets" in data:
                # Multi-sheet Excel data
                for sheet_name, sheet_data in data["sheets"].items():
                    elements.append(Paragraph(sheet_name, styles["Heading2"]))
                    elements.append(Spacer(1, 6))
                    if "headers" in sheet_data and "rows" in sheet_data:
                        elements.extend(self._pdf_table(sheet_data["headers"], sheet_data["rows"], styles))
                    elements.append(Spacer(1, 12))
            elif "full_text" in data:
                # PDF text extraction result
                for line in data["full_text"].split("\n"):
                    if line.strip():
                        elements.append(Paragraph(line, styles["Normal"]))
                        elements.append(Spacer(1, 4))
            else:
                # Generic dict
                for key, value in data.items():
                    elements.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
                    elements.append(Spacer(1, 4))
        elif isinstance(data, list):
            for item in data:
                elements.append(Paragraph(str(item), styles["Normal"]))
                elements.append(Spacer(1, 4))
        else:
            elements.append(Paragraph(str(data), styles["Normal"]))

        # Timestamp footer
        elements.append(Spacer(1, 24))
        elements.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            styles["Normal"],
        ))

        doc.build(elements)
        return buffer.getvalue(), "application/pdf"

    def _pdf_table(self, headers: List[str], rows: List[Any], styles) -> list:
        """Build a reportlab table from headers and rows."""
        from reportlab.platypus import Table, TableStyle, Spacer
        from reportlab.lib import colors

        table_data = [headers]
        for row in rows[:500]:  # Limit rows for PDF
            if isinstance(row, dict):
                table_data.append([str(row.get(h, "")) for h in headers])
            elif isinstance(row, (list, tuple)):
                table_data.append([str(c) for c in row])
            else:
                table_data.append([str(row)])

        # Truncate wide tables
        max_cols = min(len(headers), 10)
        table_data = [r[:max_cols] for r in table_data]

        t = Table(table_data)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        return [t, Spacer(1, 12)]

    def _generate_excel(self, title: str, data: Any) -> tuple:
        """Generate Excel using openpyxl."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl is required for Excel generation. Install with: pip install openpyxl")

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = title[:31]  # Excel sheet name limit

        if isinstance(data, dict):
            if "rows" in data and "headers" in data:
                self._excel_write_table(ws, data["headers"], data["rows"])
            elif "sheets" in data:
                # Multi-sheet
                first = True
                for sheet_name, sheet_data in data["sheets"].items():
                    if first:
                        ws.title = sheet_name[:31]
                        first = False
                    else:
                        ws = wb.create_sheet(title=sheet_name[:31])
                    if "headers" in sheet_data and "rows" in sheet_data:
                        self._excel_write_table(ws, sheet_data["headers"], sheet_data["rows"])
            else:
                # Key-value pairs
                ws.append(["Key", "Value"])
                for key, value in data.items():
                    ws.append([str(key), str(value)])
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                headers = list(data[0].keys())
                ws.append(headers)
                for row in data:
                    ws.append([str(row.get(h, "")) for h in headers])
            else:
                for item in data:
                    ws.append([str(item)])

        buffer = io.BytesIO()
        wb.save(buffer)
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def _excel_write_table(self, ws, headers: List[str], rows: List[Any]):
        """Write tabular data to an Excel worksheet."""
        ws.append(headers)
        for row in rows:
            if isinstance(row, dict):
                ws.append([str(row.get(h, "")) for h in headers])
            elif isinstance(row, (list, tuple)):
                ws.append([str(c) for c in row])
            else:
                ws.append([str(row)])

    def _generate_html(self, title: str, data: Any, template: Any = None) -> tuple:
        """Generate HTML document."""
        if template and isinstance(template, str):
            # Use provided HTML template
            try:
                html = template.format(title=title, data=data, timestamp=datetime.utcnow().isoformat())
            except (KeyError, IndexError):
                html = template
        else:
            # Auto-generate HTML
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #4a90d9; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.footer {{ margin-top: 2em; color: #888; font-size: 0.9em; }}
</style></head><body>
<h1>{title}</h1>
{self._data_to_html(data)}
<p class="footer">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
</body></html>"""

        return html.encode("utf-8"), "text/html"

    def _data_to_html(self, data: Any) -> str:
        """Convert data structure to HTML."""
        if isinstance(data, dict):
            if "rows" in data and "headers" in data:
                return self._table_to_html(data["headers"], data["rows"])
            elif "sheets" in data:
                parts = []
                for name, sheet in data["sheets"].items():
                    parts.append(f"<h2>{name}</h2>")
                    if "headers" in sheet and "rows" in sheet:
                        parts.append(self._table_to_html(sheet["headers"], sheet["rows"]))
                return "\n".join(parts)
            elif "full_text" in data:
                return f"<pre>{data['full_text']}</pre>"
            else:
                rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in data.items())
                return f"<table>{rows}</table>"
        elif isinstance(data, list):
            return "<ul>" + "".join(f"<li>{item}</li>" for item in data) + "</ul>"
        return f"<p>{data}</p>"

    def _table_to_html(self, headers: List[str], rows: List[Any]) -> str:
        """Render tabular data as HTML table."""
        header_html = "".join(f"<th>{h}</th>" for h in headers)
        row_html = ""
        for row in rows[:1000]:  # Limit for HTML
            if isinstance(row, dict):
                cells = "".join(f"<td>{row.get(h, '')}</td>" for h in headers)
            elif isinstance(row, (list, tuple)):
                cells = "".join(f"<td>{c}</td>" for c in row)
            else:
                cells = f"<td>{row}</td>"
            row_html += f"<tr>{cells}</tr>"
        return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"

    def validate_config(self) -> bool:
        """Validate node configuration."""
        fmt = self.config.parameters.get("format", "pdf").lower()
        if fmt not in ("pdf", "xlsx", "excel", "html"):
            raise ValueError(f"Unsupported format: {fmt}. Must be pdf, xlsx, or html")
        return True
