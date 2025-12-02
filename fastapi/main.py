import os
from fastapi import FastAPI, UploadFile, File
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKER_DIRECTORY = os.getenv("BUCKER_DIRECTORY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION   
)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Lê o conteúdo do arquivo enviado
        content = await file.read()

        # Define o caminho final dentro do bucket
        s3_key = f"{BUCKER_DIRECTORY}{file.filename}"

        # Envia para o S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=content,
            ContentType=file.content_type
        )

        return {"message": "Upload realizado com sucesso", "file": s3_key}

    except NoCredentialsError:
        return {"error": "Credenciais AWS inválidas ou ausentes."}

@app.get("/hello")
async def say_hello():
    return {"message": "Hello, world!"}