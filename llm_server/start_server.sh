# #!/bin/bash
# MODEL=${1:-"qwen2.5:1.5b"}
# echo "Starting Ollama server with model: $MODEL"
# ollama serve &
# sleep 3
# ollama pull $MODEL
# echo "Server ready at http://localhost:11434"

# #!/bin/bash
# MODEL=${1:-"qwen2.5:1.5b"}
# echo "Starting Ollama server with model: $MODEL"
# OLLAMA_NUM_CTX=4096 ollama serve
#!/bin/bash
MODEL=${1:-"gemma3:4b"}
echo "Starting Ollama server with model: $MODEL"
ollama serve &
sleep 3
ollama run $MODEL --keepalive 60m 