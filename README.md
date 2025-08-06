**Tese - Offline AI Learning Infrastructure
**
The Hub transforms Raspberry Pi devices into offline AI tutors for underserved communities. This system enables students to access curriculum-aligned education without internet connectivity.

Key Features
💡 Offline Operation: Functions without internet access

📚 Syllabus-Grounded Answers: Uses Retrieval-Augmented Generation (RAG)

📱 Lightweight Clients: Works on decade-old Android devices

☀️ Energy Efficient: 5W power consumption (solar compatible)

🤖 Multimodal Support: Text and image-based queries

Hardware Requirements
Component	Minimum Specs
Raspberry Pi	4B (4GB RAM)
Storage	32GB microSD card
Power Source	5V/3A USB-C
Client Devices	Android 5.0+ smartphones
Quick Start
1. Install Dependencies
bash
# On Raspberry Pi
sudo apt update
sudo apt install python3-pip python3-venv

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
2. Download Models
bash
ollama pull nomic-embed-text
ollama pull gemma3:7b-it-q4_K_M
3. Set Up Environment
bash
python3 -m venv edusense-env
source edusense-env/bin/activate
pip install -r requirements.txt
4. Prepare Syllabus
bash
echo "Your syllabus content here" > syllabus.txt
5. Launch Server
bash
uvicorn main:app --host 0.0.0.0 --port 8000
Client Setup
Connect to Raspberry Pi's WiFi network

Install Flutter app on Android device

Configure app to use Pi's IP address (default: 192.168.4.1:8000)

API Endpoints
Endpoint	Method	Description
/ask	POST	Submit student questions
/system-status	GET	Check server health
/update-syllabus	POST	Refresh curriculum content
Sample Question:

json
{
  "question": "Explain photosynthesis"
}
Sample Response:

json
{
  "answer": "Photosynthesis converts light to chemical energy...",
  "sources": [
    {"content": "Chapter 3 content...", "chapter": "Biology"}
  ],
  "model_used": "gemma3:7b-it-q4_K_M",
  "response_time": 1.24
}
Power Management
Switch to low-power mode automatically when battery <20%:

python
# Normal mode: 5W
# Low-power mode: 2W
Troubleshooting
Common issues and solutions:

Slow responses:

Reduce chunk size in RecursiveCharacterTextSplitter

Use smaller model: gemma3:1b-it-q4_K_M

High temperature:

Install heat sinks

Add throttle_model() in thermal protection

Connection issues:

Ensure devices on same WiFi network

Check firewall settings: sudo ufw allow 8000

License
Apache 2.0 - See LICENSE for details

Contributing
To contribute to Tese Hub:

Fork the repository

Create a new branch (git checkout -b feature)

Commit your changes (git commit -am 'Add feature')

Push to the branch (git push origin feature)

Open a pull request

