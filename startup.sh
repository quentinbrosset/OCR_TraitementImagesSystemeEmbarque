#!/bin/bash
cd /home/site/wwwroot
if [ "$APP_TYPE" = "streamlit" ]; then
  streamlit run application/app.py --server.port 8000 --server.address 0.0.0.0
else
  # Default to API (for Azure Web App)
  python -m uvicorn application.api:app --host 0.0.0.0 --port 8000
fi 