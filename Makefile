.PHONY: install test pipeline reports monitor dashboard api all
install:   ; pip install -r requirements.txt
test:      ; python -m pytest tests -q
pipeline:  ; python scripts/run_pipeline.py
reports:   ; python scripts/build_reports.py
monitor:   ; python scripts/simulate_monitoring.py
dashboard: ; streamlit run app/dashboard.py
api:       ; uvicorn readmit_audit.api:app --app-dir src --port 8000
all: pipeline reports monitor test
