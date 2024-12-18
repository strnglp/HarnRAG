import textwrap
from tabulate import tabulate
from collections import defaultdict

import rag
from color import color_text
from embedding import get_query_engine
from history import query_with_follow_up


wrapper = textwrap.TextWrapper(width=80)
while True:
    print(color_text("\n\nEnter your question ('", "yellow"), end="")
    print(color_text(">", "blue"), end="")
    print(color_text("' to follow up or '", "yellow"), end="")
    print(color_text("q", "red"), end="")
    print(color_text("' to quit): ", "yellow"), end="")
    query_str = input()
    if not query_str:
        continue
    if query_str.lower() == 'q':
        break
    response = query_with_follow_up(query_str, rag.query_engine)
    print(color_text("\nResponse:", "underline"))
    response_text = '\n'.join([
        '\n'.join(
            textwrap.wrap(
                line, 90, break_long_words=False, replace_whitespace=False
            )
        ) if line.strip() != '' else ''  # Keep blank lines intact
        for line in response.response.splitlines()
    ])
    print(color_text(response_text, "green"))
    print(color_text("\nSources:", "underline"))

    sorted_nodes = sorted(response.source_nodes, key=lambda node: node.score, reverse=True)
    book_pages = defaultdict(set)
    for node in sorted_nodes:
        metadata = node.node.metadata
        book_pages[metadata["file"]].add(metadata["page"])


    list_of_pairs = [
        [
            color_text("📘"+book, "blue"),
            color_text("📑"+", ".join(str(page) for page in pages), "magenta")
        ]
        for book, pages in book_pages.items()
    ]

    table = tabulate(list_of_pairs,
                     tablefmt="fancy_grid",
                     maxcolwidths=[60, 30],
                     colalign=("left", "left"))

    yellow = "\033[93m"
    reset = "\033[0m"
    table = table.replace(reset, yellow)
    print(color_text(table, "yellow"))
