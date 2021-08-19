WORK_DIR=$(mktemp -d)

pip install matplotlib
pip install Cython==0.28.4
pip install scikit-image==0.15.0
pip install pycocotools==2.0.0
pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git#egg=dllogger

# Install NVIDIA Apex
cd $WORK_DIR
# git clone https://github.com/NVIDIA/apex; cd apex;
# python setup.py install --cuda_ext --cpp_ext
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" git+git://github.com/NVIDIA/apex.git#egg=apex

# Install NVIDIA Data Loading Library (DALI)
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

cd /shared/DeepLearningExamples/PyTorch/Detection/SSD
pip install .

rm -rf $WORK_DIR
