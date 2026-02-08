FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
