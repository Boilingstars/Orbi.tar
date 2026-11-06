FROM python:3.12-slim

WORKDIR /app
COPY console/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY orbi /app/orbi
COPY example /app/example
COPY console /app/console
COPY reward.py observation.py task.py /app/
ENV PYTHONPATH=/app
ENV ORBI_MODEL=/models/ckpt_68120.zip
ENV ORBI_ALT_MODEL=/models/ckpt_35714.zip

EXPOSE 8000
CMD ["uvicorn", "console.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
