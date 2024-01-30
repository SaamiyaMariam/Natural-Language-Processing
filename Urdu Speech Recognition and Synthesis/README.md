# Urdu Speech Recognition and Synthesis

This project utilizes the OpenAI Whisper ASR (Automatic Speech Recognition) model for Urdu language. It performs speech recognition and synthesis on provided audio files.

## Installation

Make sure to install the required dependencies before running the code. Execute the following commands in your terminal or command prompt:

```bash
pip install -U openai-whisper
sudo apt update && sudo apt install ffmpeg
choco install ffmpeg
```
## Usage

1. **Clone the Repository:**
   - Clone this repository to your local machine.

2. **Run the Code:**
   - Place your audio files (in MP3 format) in the same directory as the `main.ipynb` notebook file.
   - Open the `main.ipynb` file using Jupyter Notebook or a compatible environment.
   - Execute the notebook cells one by one.

3. **View Results:**
   - The notebook will output the detected language and the resultant text for each audio file.

## Notes

- Ensure your audio files are in the MP3 format.
- Adjustments to the code can be made within the `main.ipynb` notebook file based on specific project requirements.
- Feel free to customize and expand upon this code for your specific use case.
- If you encounter any issues, refer to the documentation of the [OpenAI Whisper ASR API](https://platform.openai.com/docs/whisper).
