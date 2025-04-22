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

conda create --name deepsight
source activate base 
conda activate deepsight

pip install -r requirements.txt

python -m ipykernel install --user --name deepsight --display-name "deepsight"