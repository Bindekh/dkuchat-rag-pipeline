#!/bin/bash
MODEL=${1:-"qwen2.5:1.5b"}
echo "Starting Ollama server with model: $MODEL"
ollama serve &
sleep 3
ollama pull $MODEL
echo "Server ready at http://localhost:11434"
