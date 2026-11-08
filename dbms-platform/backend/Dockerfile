FROM python:3.12-slim
WORKDIR /srv/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ /srv/backend/
COPY content/ /srv/content/
ENV DBMS_CONTENT_DIR=/srv/content PYTHONUNBUFFERED=1
RUN useradd -r app && chown -R app /srv
USER app
EXPOSE 8000
# One worker: concurrent lab sessions (A/B/C) are held in-process. Scale out with sticky sessions.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers"]
