Task 2 – Person B Deliverables

- Created a Dockerfile using python:3.9-slim, copied project files into /app, installed pandas, matplotlib, seaborn, and ran data_analysis.py.

- Built image: docker build -t diet-analysis .

- Verified container execution: docker run --rm -it diet-analysis

- Persisted outputs to host using volume mapping: docker run --rm -it -v "${PWD}\outputs:/app/outputs" diet-analysis

- Simulated deployment using Docker Compose: docker compose up --build and confirmed successful exit (code 0).