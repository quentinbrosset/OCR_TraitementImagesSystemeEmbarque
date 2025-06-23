#!/bin/bash
cd /home/site/wwwroot
if [ "$APP_TYPE" = "api" ]; then
  python -m uvicorn application.api:app --host 0.0.0.0 --port 8000
else
  streamlit run application/app.py --server.port 8000 --server.address 0.0.0.0
fi 