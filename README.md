# slang-aware-hate-speech

The primary research problem we aim to address is the lack of nuance in current models used to classify social media comments as hate speech. Existing systems often struggle to adapt to the evolving nature of internet slang and emoji use, leading to frequent misclassifications. These models often rely on static dictionaries or predefined corpora which fail to capture the fluidity and content of online language. This is particularly evident when dealing with modern slang, where words and symbols can carry different meanings depending on context, making hate speech detection more complex.

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/kalefed/slang-aware-hate-speech.git

   ```

2. Navigate to project dirctory

   ```bash
   cd slang-aware-hate-speech

   ```

3. Set up the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies
    ```bash
    pip install -r requirements.txt

5. Run the project
    ```bash
    python src/main.py