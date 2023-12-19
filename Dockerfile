# Stage 1: Python Flask app
FROM python:3.9-slim

# Mengatur working directory di dalam container
WORKDIR /app

# Menyalin file API.py ke dalam working directory
COPY API.py /app/API.py

# Instalasi Flask secara langsung di dalam container
RUN pip install flask

# Menjalankan Flask app ketika container dimulai
CMD ["python", "API.py"]