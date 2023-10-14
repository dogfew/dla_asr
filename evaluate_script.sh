# install requirements
pip install -r requirements.txt
# create directory
pip install gdown>4.7
mkdir -p default_test_model
cd default_test_model
gdown 1Pjhw3YC991OPCTdSIcAsO3seHqhvkXR_ -O checkpoint.pth
gdown 1JrK51xkfYZzWZJf9INFyqTIVDQmiiChJ -O config.json
gdown 13UDHWNckiFJtKFHucmM7gewddikWSFh1 -O test_config.json
cd ..
