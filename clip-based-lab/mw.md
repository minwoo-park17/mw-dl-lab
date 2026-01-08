# 1. Conda 환경 생성 (Python 3.10)
conda create -n clip-lab python=3.10 -y
conda activate clip-lab

# 2. PyTorch + CUDA 11.8 설치 (반드시 conda로 먼저!)
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 나머지 패키지 설치
pip install -r requirements.txt