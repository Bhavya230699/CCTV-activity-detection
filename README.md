\# CCTV Activity Detection



A computer vision application for real-time activity, motion, and anomaly detection in CCTV feeds.



\## 📂 Project Structure

\- `app.py`: Flask application server and routes.

\- `live\_monitor.py`: Video stream processor and real-time detection pipeline.

\- `main.py`: Main entry point for execution.

\- `models/`: Neural network architecture and model registry files.

\- `templates/` \& `static/`: Web dashboard frontend interfaces.



\## 📊 Dataset Access

Due to GitHub's file size limits, the training and validation dataset is hosted externally.



👉 \*\*\[Click Here to Download the CCTV Activity Dataset](https://drive.google.com/drive/folders/1KK47lmP94YtTLW\_SSf9wZwwBILHVmhaO?usp=sharing)\*\*



\## 🚀 How to Run

1\. Install dependencies:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

Download and place the dataset inside a datasets/ folder in the project root.

Start the application:

Bash
python app.py
