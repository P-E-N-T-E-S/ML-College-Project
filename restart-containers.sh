#!/bin/bash
echo "🔄 Parando containers..."
docker-compose down
echo "✨ Iniciando containers com emulação Rosetta..."
docker-compose up -d
echo "✅ Containers iniciados!"
echo "📊 Status dos containers:"
docker-compose ps
