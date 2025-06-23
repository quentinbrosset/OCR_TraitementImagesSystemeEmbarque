# startup.py - Script de démarrage pour Azure Web App
import os
import sys
import subprocess

def main():
    # Ajouter le répertoire de l'application au path
    app_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, app_dir)
    
    # Déterminer quel type d'application démarrer
    app_type = os.environ.get('APP_TYPE', 'streamlit')
    port = int(os.environ.get('PORT', 8000))
    
    if app_type == 'api':
        print("Starting FastAPI application...")
        # Démarrer l'API FastAPI
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "application.api:app", 
            "--host", "0.0.0.0", 
            "--port", str(port)
        ]
    else:
        print("Starting Streamlit application...")
        # Démarrer l'application Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "application/app.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
    
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

# requirements.txt optimisé pour Azure
requirements_content = """
fastapi==0.104.1
uvicorn[standard]==0.24.0
streamlit==1.28.1
torch==2.1.1
torchvision==0.16.1
Pillow==10.1.0
numpy==1.24.3
opencv-python-headless==4.8.1.78
pandas==2.1.3
matplotlib==3.8.2
requests==2.31.0
python-multipart==0.0.6
"""

# .streamlit/config.toml - Configuration Streamlit
streamlit_config = """
[server]
headless = true
port = 8000
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[logger]
level = "info"
"""
