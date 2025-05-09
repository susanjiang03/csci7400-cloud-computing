## Cloud-Based Vision Transformer for Pathological Myopia Detection

### AWS SageMaker AI notebook configuration

##### To enable Lambda function trigger notebook execution, the notebook instance need to be configured by following steps


* Create virtual enviroment:

List env list
```
conda env list
```

Create and activate a virtual env
````
conda create --name deepsight
source activate base 
conda activate deepsight
````

Create requirements file in notebook instance
```
cat > requirements.txt << EOF 
numpy
pandas
matplotlib
torch
pytorch-lightning
torchvision
torchinfo
torchview
graphviz
IPython
lightning
boto3
requests==2.31.0
ws4py
urllib3==1.26.12
openpyxl
ipykernel
EOF
```

Create a file of requirements.txt and upade the content
Install all packages:
```
pip install -r requirements.txt
````

Register a kernel
```
python -m ipykernel install --user --name deepsight --display-name "deepsight"
```

Execute a notebook script from notebook temrminal:
````
jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/PTViT_NAM.ipynb --ExecutePreprocessor.kernel_name=deepsight --ExecutePreprocessor.timeout=1500
````

with variables:
````
SIZE=100 BATCH_SIZE=2 jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/final.ipynb --ExecutePreprocessor.kernel_name=deepsight --ExecutePreprocessor.timeout=1500
````


##### Deploying Lambda functions as .zip file archives

* Creating a .zip deployment package with dependencies

cd lambdaf_function_trigger
python3 -m venv myvirtualenv
source myvirtualenv/bin/activate

pip install --target ./package -r requirements.txt

zip -r package.zip package lambda_function.py

* Upload the package.zip to update package dependencies