import json
from websocket import create_connection
import boto3
import time
import requests

def lambda_handler(event, context):
    sm_client = boto3.client('sagemaker')
    notebook_instance_name = 'deepsight-test3'
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

    ws.send("""[ "stdin", "jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/first.ipynb --ExecutePreprocessor.kernel_name=conda_python3 --ExecutePreprocessor.timeout=1500\\r" ]""")
    print (ws.recv())
    ws.close()