FROM vllm/vllm-openai:latest

# transformers 최신화(OPTIONAL, vLLM과 호환되는 버전만 허용)
RUN pip install --no-cache-dir --upgrade "transformers>=4.41.0"

# 모델을 /model에 마운트할 것이므로 작업 디렉토리만 설정
WORKDIR /model

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/model", "--host", "0.0.0.0", "--port", "8000"]
