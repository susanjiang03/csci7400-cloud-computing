

conda env list

Create a virtual env
````
conda create --name deepsight
source activate base 
conda activate deepsight
````

create a file of requirements.txt and upade the content
Install all packages:
```
pip install -r requirements.txt
````

To register a kernel
```
python -m ipykernel install --user --name deepsight --display-name "deepsight"
```

To execute a notebook file from temrminal:
````
jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/PTViT_NAM.ipynb --ExecutePreprocessor.kernel_name=deepsight --ExecutePreprocessor.timeout=1500
````

with variables:
````
SIZE=100 BATCH_SIZE=2 jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/PTViT_NAM.ipynb --ExecutePreprocessor.kernel_name=deepsight --ExecutePreprocessor.timeout=1500
````