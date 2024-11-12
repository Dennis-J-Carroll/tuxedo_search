import tensorflow as tf

def preprocess_text(text):
    # Example preprocessing: Convert to lowercase and tokenize
    text = text.lower()
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    return tokens
