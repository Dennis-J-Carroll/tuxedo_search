# Import necessary libraries and modules
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import preprocess_text
from postprocessing import decode_output

# Load the trained model
model = load_model('DeNN_o1.h5')

model.save('DeNN_o1.h5')

def chat_with_model():
    print("Welcome to the Neural Network Chatbot!")
    print("Type 'quit' to exit the chat.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)

        # Generate a response from the model
        response = model.predict(preprocessed_input)
        decoded_response = decode_output(response)

        print("Chatbot:", decoded_response)

# Start the interactive chat
chat_with_model()
