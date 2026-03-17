import re


class TableToJSON:
    def convert(self, rows):
        rows = self._normalize(rows)
        if len(rows) < 2:
            return {}

        headers = self._get_unique_headers(rows)
        table_json = {}

        for row in rows[1:]:
            raw_key = row[0].strip()
            row_key = raw_key if raw_key else f"Item_{rows.index(row)}"

            if row_key in table_json:
                row_key = f"{row_key}_{rows.index(row)}"

            table_json[row_key] = {}

            for col, val in zip(headers[1:], row[1:]):
                clean_val = self._clean(val)
                if clean_val and clean_val not in ["-", "—", "N/A", "null"]:
                    table_json[row_key][col] = clean_val

        return table_json

    def _normalize(self, rows):
        if not rows: return []
        max_len = max(len(r) for r in rows)
        return [list(r) + [""] * (max_len - len(r)) for r in rows]

    def _get_unique_headers(self, rows):
        """Extracts headers and ensures they are unique and non-empty."""
        raw_headers = rows[0]
        for r in rows:
            if sum(1 for c in r if c.strip()) > 1:
                raw_headers = r
                break

        unique_headers = []
        for i, h in enumerate(raw_headers):
            h_clean = h.strip() or (f"Column_{i}" if i > 0 else "Metric")
            if h_clean in unique_headers:
                h_clean = f"{h_clean}_{i}"
            unique_headers.append(h_clean)
        return unique_headers

    def _clean(self, val):
        """Removes citations [1], commas in numbers, and extra whitespace."""
        if not val: return ""
        val = re.sub(r"[\*\_]", "", val)
        val = re.sub(r"\[\d+\]", "", val)
        if re.match(r"^\d{1,3}(,\d{3})*(\.\d+)?$", val.strip()):
            val = val.replace(",", "")
        return val.strip()
