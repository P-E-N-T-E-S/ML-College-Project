#carregar variaveis de ambiente do arquivo .env

export $(grep -v '^#' .env | xargs)

#subir os containers necessarios
docker-compose up -d --build