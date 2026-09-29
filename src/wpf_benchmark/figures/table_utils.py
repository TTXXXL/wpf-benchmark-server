"""booktabs LaTeX 和 Markdown 表格共用写入。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence


def escape_tex(value: object) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def write_table(out: Path, stem: str, headers: Sequence[str],
                rows: Sequence[Sequence[str]], alignment: str,
                note: str = "") -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    if len(alignment) != len(headers):
        raise ValueError("LaTeX alignment width differs from columns")
    tex = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
           " & ".join(headers) + r" \\", r"\midrule"]
    tex += [" & ".join(row) + r" \\" for row in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    if note:
        tex += [r"\par\footnotesize{" + escape_tex(note) + "}"]
    md = ["| " + " | ".join(header.replace(r"\_", "_").replace(r"\%", "%")
                              for header in headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"]
    md += ["| " + " | ".join(cell.replace(r"\textbf{", "**").replace("}", "**")
                                for cell in row) + " |" for row in rows]
    if note:
        md += ["", note]
    outputs = [out / (stem + ".tex"), out / (stem + ".md")]
    outputs[0].write_text("\n".join(tex) + "\n", encoding="utf-8")
    outputs[1].write_text("\n".join(md) + "\n", encoding="utf-8")
    return outputs
