# michaelf34/infinity:0.0.77-cpu ships transformers==4.49.0.dev0, which predates
# qwen3 architecture support (added in 4.51.0). infgrad/Jasper-Token-Compression-600M
# requires transformers>=4.57.1. Pinned (not "latest") because transformers 5.x
# breaks the image's bundled sentence-transformers==3.3.1, which requires <5.0.0.
FROM michaelf34/infinity:0.0.77-cpu

RUN pip install --no-cache-dir "transformers==4.57.1"
