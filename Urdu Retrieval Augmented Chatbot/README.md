# Urdu Retrieval Augmented Chatbot

The project aims to build an interactive conversational agent fluent in Urdu, leveraging the Retrieval Augmented Generation (RAG) model. The integration of Speech-to-Text (STT) and Text-to-Speech (TTS) modules further enriches the bot's capabilities, enabling it to comprehend spoken Urdu and respond coherently using synthesized speech.

## Project Files
- `I20_0612.ipynb`

## Installation
Make sure to install the required dependencies before running the code. Execute the following commands in your terminal or command prompt:

```bash
pip install SpeechRecognition
pip install pydub gtts playsound datasets
pip install accelerate -U
pip install --upgrade accelerate
```
## Code Overview

The notebook `I20_0612.ipynb` includes the following functionality:

- Cleaning text data
- Speech-to-text conversion
- Text-to-speech synthesis
- Loading and processing the dataset
- Training the RAG model
- Fine-tuning and saving the model
- Implementing the RAG Bot function

## Usage

1. **Clone the Repository:**
   - Clone this repository to your local machine.

2. **Run the Code:**
   - Open the `I20_0612.ipynb` notebook using Jupyter Notebook or a compatible environment.
   - Execute the notebook cells one by one.

3. **View Results:**
   - The notebook includes a sample usage of the RAG Bot function with an audio file named "audio_file.mp3". Modify as needed.

## Notes

- Ensure your audio files are in the MP3 format.
- If you encounter any issues, refer to the documentation of the [OpenAI Whisper ASR API](https://platform.openai.com/docs/whisper).
