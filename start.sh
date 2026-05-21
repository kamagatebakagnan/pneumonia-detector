#!/bin/bash
# Lancer FastAPI en arrière-plan
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Lancer Streamlit au premier plan
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
