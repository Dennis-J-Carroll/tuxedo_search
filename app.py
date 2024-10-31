from flask import Flask, request, render_template

app = Flask(__name__)

# Sample documents to search
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a great programming language",
    "Flask is a micro web framework for Python",
    "Search engines are useful for finding information"
]

def search_documents(query):
    """Simple search function to match query against documents."""
    results = [doc for doc in documents if query.lower() in doc.lower()]
    return results

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query', '')
        results = search_documents(query)
        return render_template('results.html', query=query, results=results)
    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)
