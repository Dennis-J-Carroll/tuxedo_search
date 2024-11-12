import os
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Sample documents to search
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a great programming language",
    "Flask is a micro web framework for Python",
    "Search engines are useful for finding information"
]

def search_documents(query):
    """Search function using cosine similarity."""
    vectorizer = TfidfVectorizer().fit_transform([query] + documents)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors[0:1], vectors[1:])
    results = [documents[i] for i in cosine_matrix.argsort()[0][::-1] if cosine_matrix[0][i] > 0]
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
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/train', methods=['POST'])
def train():
    if 'dataset' not in request.files:
        return jsonify({'error': 'No dataset file provided.'}), 400

    dataset = request.files['dataset']
    epochs = int(request.form['epochs'])

    if not dataset:
        return jsonify({'error': 'Invalid dataset file.'}), 400

    filename = secure_filename(dataset.filename)
    dataset.save(filename)

    model = None

    @app.before_first_request
    def load_model():
        global model
        model = keras.models.load_model('model.h5')
    
    history = model.fit(dataset, epochs=epochs)
    model.save('model.h5')

    os.remove(filename)

    return jsonify({
        'epochs': [i for i in range(epochs)], 
        'loss': history.history['loss'],
        'summary': model.summary()
    })
