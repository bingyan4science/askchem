FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/

ENV ASKCHEM_DB=/app/data/askchem.db
ENV PYTHONPATH=/app/src
ENV CHEMTREE_DISABLE_PAW=1

EXPOSE 8080

CMD ["uvicorn", "askchem.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
