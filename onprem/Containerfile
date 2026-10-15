# Deterministic root-cause service. UBI base, no GPU, no model weights, no egress.
FROM registry.access.redhat.com/ubi9/python-312:latest
USER 1001
WORKDIR /opt/app-root/src
RUN pip install --no-cache-dir "numpy>=1.26"
COPY rca_service/ ./rca_service/
COPY topology/ ./topology/
ENV TOPOLOGY=/opt/app-root/src/topology/ran_core_transport.json \
    PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python3 -c \
  "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=2).status==200 else 1)"
ENTRYPOINT ["python3", "-m", "rca_service.server"]
