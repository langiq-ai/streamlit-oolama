
# Streamlit Ollama Interaction App
This project is a Streamlit application that interacts with the Ollama language model. The app allows users to input a message and receive a response generated by the Ollama model.

## Features

- **User Input**: Users can enter their messages through a text input field.
- **Ollama Model**: Utilizes the Ollama language model to generate responses.
- **Logging**: Detailed logging of events and data for debugging and monitoring purposes.

## Installation

1. **Clone the repository**:
    ```sh
    git git@github.com:langiq-ai/streamlit-oolama.git
    cd streamlit-oolama
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Interact with the app**:
    - Enter your message in the text input field on the sidebar.
    - Click the "Send" button to get a response from the Ollama model.

## Code Overview

### `app.py`

This is the main file of the application. It contains the following key components:

- **Logging Setup**: Configures the logging module to log events and data.
- **Streamlit UI**: Sets up the Streamlit interface, including the title and sidebar.
- **Prompt Template**: Defines the template for the prompt sent to the Ollama model.
- **User Input Handling**: Captures user input and processes it.
- **Ollama Model Initialization**: Initializes the Ollama model.
- **Response Generation**: Uses the `LLMChain` to generate a response based on user input.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.  
### License
This project is licensed under the MIT License. See the LICENSE file for more details.  
#### Contact
For any questions or inquiries, please contact sandesh@langiq.ai 

This `README.md` file provides a comprehensive overview of the project, including installation instructions, usage, code overview, and contribution guidelines.
