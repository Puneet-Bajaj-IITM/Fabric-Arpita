
# Project Setup Guide

Follow the steps below to set up your environment:

1. **Create or Attach to a `tmux` Session**

   If you don't have a session running, create one:

   ```bash
   tmux new-session -s your_session_name
   ```

   If the session already exists, attach to it:

   ```bash
   tmux attach -t your_session_name
   ```

2. **Install Virtualenv (if not already installed)**

   ```bash
   pip install virtualenv
   ```

3. **Create and Activate a Virtual Environment**

   Create a new virtual environment:

   ```bash
   virtualenv venv
   ```

   Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

4. **Install System Dependencies (for Ubuntu/Debian)**

   Install `libzbar0` (for barcode scanning):

   ```bash
   sudo apt update
   sudo apt install libzbar0
   ```

   Install SQLite3:

   ```bash
   sudo apt install sqlite3
   ```

5. **Install Required Python Packages**

   Install all necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Application**

   Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   The app will be available on port `8501`. To detach from `tmux` and keep the app running, press `Ctrl + B`, then `D`.

7. **Visit the App**

   Open your browser and visit `http://localhost:8501` to check the app.

Enjoy using your app!
