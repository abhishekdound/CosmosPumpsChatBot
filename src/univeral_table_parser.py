from bs4 import BeautifulSoup


class UniversalTableParser:
    def parse_html(self, html):
        """Parses HTML tables, filtering out nested table noise and empty rows."""
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        parsed_tables = []

        for table in tables:
            rows = []
            for tr in table.find_all("tr", recursive=False):
                cells = [
                    td.get_text(" ", strip=True)
                    for td in tr.find_all(["td", "th"], recursive=False)
                ]
                if any(cells):
                    rows.append(cells)

            if len(rows) >= 2:
                parsed_tables.append(rows)

        return parsed_tables

    def parse_markdown(self, md):
        """Parses Markdown pipe tables, explicitly ignoring syntax separators."""
        if not md:
            return []

        lines = [line.strip() for line in md.split("\n") if line.strip()]
        all_tables = []
        current_table = []

        for line in lines:
            if "|" in line:
                if all(char in "| -:" for char in line):
                    continue

                row = [cell.strip() for cell in line.split("|")]

                if line.startswith("|"): row.pop(0)
                if line.endswith("|"): row.pop()

                if any(row):
                    current_table.append(row)
            else:
                if len(current_table) >= 2:
                    all_tables.append(current_table)
                current_table = []

        if len(current_table) >= 2:
            all_tables.append(current_table)

        return all_tables


if __name__ == "__main__":
    parser = UniversalTableParser()

