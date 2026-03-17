from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


def LLMTableToJson(llm, table_text):
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template("""
    Convert the following raw table text into a structured JSON object.

    Structure requirements:
    - Use a 'header-to-value' mapping.
    - The top-level keys should represent the primary entity/metric (the first column).
    - Nested keys should be the column headers.

    Example Format:
    {{
      "Pump Model A": {{"Flow Rate": "50m3/h", "Power": "10kW"}},
      "Pump Model B": {{"Flow Rate": "75m3/h", "Power": "15kW"}}
    }}

    Rules:
    - Return ONLY the JSON object.
    - Do not include markdown code blocks (e.g., ```json).
    - If the table is empty, return an empty object {{}}.

    Table:
    {table}
    """)

    chain = prompt | llm | parser

    try:
        return chain.invoke({"table": table_text})
    except OutputParserException as e:
        print(f"Failed to parse table JSON: {e}")
        return {}
