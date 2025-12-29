import re
from pathlib import Path
from typing import List, Optional, Tuple


def remove_bu_wrappers(text: str) -> str:
    r"""Remove only \mathbf{...}, \underline{...}, and \textbf{...} macros, preserving everything else."""
    pattern = re.compile(r"\\(?:mathbf|underline|textbf)\{([^{}]*)\}")
    prev = None
    out = text
    while prev != out:
        prev = out
        out = pattern.sub(r"\1", out)
    return out


def math_inner(cell: str) -> Tuple[bool, str]:
    s = cell.strip()
    m = re.fullmatch(r"\$\s*(.*?)\s*\$", s)
    if m:
        return True, m.group(1)
    return False, cell


def set_math_inner(original: str, new_inner: str, had_math: bool) -> str:
    if had_math:
        # Preserve leading/trailing spaces outside math if any
        leading = "" if not original.startswith(" ") else original[: len(original) - len(original.lstrip(" "))]
        trailing = "" if not original.endswith(" ") else original[len(original.rstrip(" ")):]
        return f"{leading}${new_inner}${trailing}" if (leading or trailing) else f"${new_inner}$"
    return original


def extract_value(cell: str) -> Optional[float]:
    had_math, inner = math_inner(cell)
    # Only consider numeric values inside math cells to avoid picking up header text like 'F1' or 'Hd95'.
    if not had_math:
        return None
    cleaned = remove_bu_wrappers(inner)
    # Try pattern with \pm
    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*\\pm", cleaned)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # Try plain float
    m2 = re.search(r"([-+]?[0-9]*\.?[0-9]+)", cleaned)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            return None
    return None


def split_row(line: str) -> Tuple[List[str], str]:
    term_match = re.search(r"\\\\\s*$", line)
    terminator = term_match.group(0) if term_match else ""
    body = line[: term_match.start()] if term_match else line
    cells = [c.strip() for c in body.split("&")]
    return cells, terminator


def join_row(cells: List[str], terminator: str) -> str:
    return " & ".join(cells) + terminator


def find_tabular_blocks(content: str) -> List[Tuple[int, int]]:
    blocks = []
    for m in re.finditer(r"\\begin\{tabular\}[^\n]*\n", content):
        start = m.start()
        end_m = re.search(r"\\end\{tabular\}", content[m.end():])
        if end_m:
            end = m.end() + end_m.end()
            blocks.append((start, end))
    return blocks


def process_tabular(tab: str) -> str:
    lines = tab.splitlines()
    row_indices = [i for i, ln in enumerate(lines) if '&' in ln and re.search(r"\\\\\s*$", ln)]
    if not row_indices:
        return tab

    sample_cells, _ = split_row(lines[row_indices[0]])
    col_count = len(sample_cells)

    numeric_start_col = 2  # from the structure: Method, Model, then metrics
    numeric_cols = list(range(numeric_start_col, col_count))

    # Columns where lower values are better: Hd95 (8), Betti 0 Error (9), Betti 1 Error (10)
    lower_better_cols = [8, 9, 10]

    # First pass: compute values and strip bu wrappers in-memory only
    column_values = {j: [] for j in numeric_cols}
    for i in row_indices:
        cells, _ = split_row(lines[i])
        if len(cells) != col_count:
            continue
        for j in numeric_cols:
            v = extract_value(cells[j])
            if v is not None:
                column_values[j].append((i, v))

    # Determine ranks explicitly for clarity: highest/second-highest for higher-better,
    # lowest/second-lowest for lower-better; ties allowed
    col_ranks = {}
    for j in numeric_cols:
        vals = [v for _, v in column_values[j]]
        if not vals:
            continue
        if j in lower_better_cols:
            ordered = sorted(set(vals))
        else:
            ordered = sorted(set(vals), reverse=True)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else None
        col_ranks[j] = (top, second)

    # Second pass: rewrite only \mathbf/\underline wrappers inside math cells
    for i in row_indices:
        cells, term = split_row(lines[i])
        if len(cells) != col_count:
            lines[i] = join_row(cells, term)
            continue
        for j in range(col_count):
            # Remove existing wrappers only for numeric columns; keep others untouched
            if j in numeric_cols:
                had_math, inner = math_inner(cells[j])
                inner_clean = remove_bu_wrappers(inner)
                action = None
                if j in col_ranks:
                    v = extract_value(cells[j])
                    if v is not None:
                        top, second = col_ranks[j]
                        if v == top:
                            action = 'mathbf'
                        elif second is not None and v == second:
                            action = 'underline'
                if action:
                    if had_math:
                        inner_new = f"\\{action}{{{inner_clean}}}"
                        cells[j] = set_math_inner(cells[j], inner_new, had_math=True)
                    else:
                        # Apply wrappers even if the cell isn't in math mode
                        if action == 'mathbf':
                            cells[j] = f"\\textbf{{{remove_bu_wrappers(cells[j])}}}"
                        elif action == 'underline':
                            cells[j] = f"\\underline{{{remove_bu_wrappers(cells[j])}}}"
                else:
                    # If no action, still ensure we removed only bu wrappers; keep math as-is
                    if had_math:
                        cells[j] = set_math_inner(cells[j], inner_clean, had_math=True)
                    else:
                        cells[j] = remove_bu_wrappers(cells[j])
        lines[i] = join_row(cells, term)

    return "\n".join(lines)


def reformat_file(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    blocks = find_tabular_blocks(text)
    if not blocks:
        # Fallback: just strip and do nothing else (only modify bu wrappers)
        cleaned = remove_bu_wrappers(text)
        path.write_text(cleaned, encoding='utf-8')
        return

    out = []
    last = 0
    for (s, e) in blocks:
        out.append(text[last:s])
        out.append(process_tabular(text[s:e]))
        last = e
    out.append(text[last:])
    new_text = "".join(out)
    path.write_text(new_text, encoding='utf-8')


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Reapply LaTeX table styling: remove only \\mathbf/\\underline then "
            "bold maxima and underline second-highest per numeric column."
        )
    )
    parser.add_argument("file", help="Path to the .tex file to process")
    args = parser.parse_args()

    tex_path = Path(args.file)
    if not tex_path.exists():
        raise SystemExit(f"File not found: {tex_path}")

    reformat_file(tex_path)
    print(f"Updated: {tex_path}")


if __name__ == "__main__":
    main()
