import os
import sys
import re
import markdown
import logging
from flask import render_template
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from tabulate import tabulate
from collections import defaultdict


# too lazy to organize my project right now
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rag
from color import color_text
from embedding import get_query_engine
from history import query_with_follow_up


app = Flask(__name__)

app.config['BASIC_AUTH_USERNAME'] = 'harnmaster'
app.config['BASIC_AUTH_PASSWORD'] = 'whatisjmorvi'
app.config['BASIC_AUTH_FORCE'] = True 

basic_auth = BasicAuth(app)

log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

app.logger.handlers = logging.getLogger().handlers
app.logger.setLevel(logging.INFO)
@app.before_request
def log_request_info():
    logging.info(f"Request: {request.method} {request.path}")
    logging.info(f"Headers: {dict(request.headers)}")
    logging.info(f"Body: {request.get_data().decode('utf-8')}")
    
@app.route('/')
def index():
    return render_template('index.html')


def process_input(query_str):
    response = query_with_follow_up(query_str, rag.query_engine)
    response_text = markdown.markdown(response.response, extensions=['tables'])
    pattern = r'^(\s*<p)(.*>)\s*•'  # Capture the opening <p> tag and its closing '>'
    replacement = r"\1 class='no-indent'\2•"  # Add class inside the <p> tag
    response_text = re.sub(pattern, replacement, response_text, flags=re.MULTILINE)
    
    sorted_nodes = sorted(response.source_nodes, key=lambda node: node.score, reverse=True)
    book_pages = defaultdict(set)
    for node in sorted_nodes:
        metadata = node.node.metadata
        book_pages[metadata["book"]].add(metadata["page"])
        print(f"{metadata} - score: {node.score}")

    list_of_pairs = [
        [
            book,
            ", ".join(str(page) for page in pages)
        ]
        for book, pages in book_pages.items()
    ]

    table_html = "<table>"
    table_html += "<thead><tr><th>Book</th><th>Pages</th></tr></thead>"
    table_html += "<tbody>"

    for row in list_of_pairs:
        table_html += f"<tr><td>{row[0]}</td><td>{row[1]}</td></tr>"

    table_html += "</tbody></table>"

    return f"""
    {response_text}
    {table_html}
    """


@app.route('/process', methods=['POST'])
def process():
    user_input = request.json.get('input', '')
    colored_input = color_text(user_input, "yellow")
    logging.info(f"Processing input: {colored_input}")
    output = process_input(user_input)
    colored_output = color_text(output, "magenta")
    logging.info(f"Sending output: {colored_output}")
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

