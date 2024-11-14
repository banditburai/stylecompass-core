import lancedb
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def init_db():
    bucket_name = os.getenv('AWS_BUCKET_NAME')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    endpoint = os.getenv('AWS_ENDPOINT')
    
    if not all([bucket_name, aws_access_key_id, aws_secret_access_key, endpoint]):
        raise ValueError("Missing required configuration. Please check your .env file.")

    db_path = f"s3://{bucket_name}/lancedb"
    
    storage_options = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "endpoint": endpoint,
        "region": "us-east-1",
        "allow_http": "true"
    }
    
    db = await lancedb.connect_async(db_path, storage_options=storage_options)
    return db 