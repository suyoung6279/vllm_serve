FROM vllm/vllm-openai:gemma4

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV MODEL_NAME="cyankiwi/gemma-4-12B-it-AWQ-INT4"
ENV VLLM_BASE_URL="http://127.0.0.1:8000/v1"

WORKDIR /app

COPY requirements_vllm.txt .
RUN pip install --no-cache-dir -r requirements_vllm.txt

COPY server_vllm.py .
COPY start.sh .
RUN chmod +x /app/start.sh

# vLLM: 8000 / FastAPI: 8501
EXPOSE 8000 8501

CMD ["/app/start.sh"]
