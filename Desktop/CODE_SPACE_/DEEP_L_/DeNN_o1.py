# Existing code for the neural network model goes here

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
