from minio import Minio
import os
from dotenv import load_dotenv
load_dotenv()

client = Minio(
    os.getenv("MINIO_URL_INTERNAL").replace("http://", ""),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)

def upload_to_minio(file, bucket):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    client.put_object(
        bucket, file.filename, file.file, length=-1, part_size=20*1024*1024
    )
    url = f"{os.getenv('MINIO_URL_EXTERNAL')}/{bucket}/{file.filename}"
    return url