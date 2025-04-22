import json
from websocket import create_connection
import boto3
import time
import requests
import os

s3_cient = boto3.client('s3')
sm_client = boto3.client('sagemaker')
notebook_instance_name = 'deepsight'
notebook_script_name = "final.ipynb"
kernel_name="deepsight"
TIMEOUT = os.environ.get("TIMEOUT", -1)
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", 100)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 100)
MAX_EPOCHS = os.environ.get("MAX_EPOCHS", 2)

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print(f"s3 {bucket} Received file: {key}")
    prefix = "PALM-Validation400/"
    count = 0
    paginator = s3_cient.get_paginator('list_objects')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page['Contents']:
                result = s3_cient.head_object(Bucket=bucket, Key=obj['Key'])
                count += 1
    print(f"number of picture in folder {prefix}: {count}")
    # only invoke notebook script for every 5 new pictures
    # if count%5 != 0:
    #     return

    print("Start training")
    url = sm_client.create_presigned_notebook_instance_url(NotebookInstanceName=notebook_instance_name)['AuthorizedUrl']
    url_tokens = url.split('/')
    http_proto = url_tokens[0]
    http_hn = url_tokens[2].split('?')[0].split('#')[0]

    s = requests.Session()
    r = s.get(url)
    cookies = "; ".join(key + "=" + value for key, value in s.cookies.items())

    ws = create_connection(
        f"wss://{http_hn}/terminals/websocket/1",
        cookie=cookies,
        host=http_hn,
        origin=f"{http_proto}//{http_hn}"
    )
    
    enviro_varible_string=f"MAX_EPOCHS={MAX_EPOCHS} BATCH_SIZE={BATCH_SIZE}"
    command_string = f"{enviro_varible_string} nohup jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/{notebook_script_name} --ExecutePreprocessor.kernel_name={kernel_name} --ExecutePreprocessor.timeout={TIMEOUT}\\r"
    print(f"Sending command to sagemaker notebook instance {notebook_instance_name}: {command_string}")
    ws.send(f"""[ "stdin", "{command_string}" ]""")
    print (ws.recv())
    ws.close()