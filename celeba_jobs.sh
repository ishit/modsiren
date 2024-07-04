#!/bin/bash
echo "Starting worker $1 on $(hostname)"
echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo "nvidia-smi failed, aborting"
    echo $(hostname)
    exit
fi

# Set root dirs
# ROOT=/home/ishit/temp_root/
# OUTPUT=/home/ishit/temp_out/
DATA=/mnt/ilcompf9d1/user/mgharbi/data/ishit
ROOT=/mnt/ilcompf9d1/user/mgharbi/code/ishit
OUTPUT=/mnt/ilcompf9d1/user/mgharbi/output/ishit

# # get code
# cd $ROOT

# if [ ! -d $ROOT"/localImplicit_release" ]
# then
#     git clone https://github.com/ishit/localImplicit_release.git
# else
#     cd $ROOT"/localImplicit_release"
#     git pull
#     cd ..
# fi

# move code and data to shared memory
CODE=$ROOT
CODE_LOCAL=/dev/shm/$1/ishit
mkdir -p $CODE_LOCAL

echo "Copying code and data to local memory drive"
echo $CODE $(dirname $CODE_LOCAL)
mkdir -p /dev/shm/$1
rsync -av --ignore-existing $CODE $(dirname $CODE_LOCAL)

# install python modules
cd $CODE_LOCAL
echo "current dir" $(pwd)
echo "Installing python modules"
export PYTHONPATH=.:$PYTHONPATH
pip install -r requirements.txt
pip install lpips # didn't worky locally with requirements.txt!

# get config scripts
CONFIG_DIR=$CODE_LOCAL/config/img_gen/celeba

echo $CONFIG_DIR
configs=()
for c in "$CONFIG_DIR"/*.yml
do
    configs+=($c)
done

let jobid=$1
curr_config="${configs[$jobid]}"
echo "Running job" $jobid on worker $1 with config $curr_config

cd $CODE_LOCAL

# get data
echo "downloading data"
# wget -nc http://132.239.95.124:8000/celeba.tar.gz 
rsync -av --ignore-existing $DATA/celeba.tar.gz  .

echo extracting data
if [ ! -d "img_align_celeba" ] 
then
    tar zvxf celeba.tar.gz
fi

echo config: $curr_config
echo Training
mkdir -p $OUTPUT

# edit config file
cd $CODE_LOCAL
cp $curr_config temp_config.yml

LOG_DIR=$OUTPUT/logs_celeba
train_filenames=$CODE_LOCAL/img_align_celeba/train_filenames.txt
val_filenames=$CODE_LOCAL/img_align_celeba/val_filenames.txt
test_filenames=$CODE_LOCAL/img_align_celeba/test_filenames.txt
root_dir=$CODE_LOCAL/img_align_celeba/
log_dir=$CODE_LOCAL/img_align_celeba/

echo $val_filenames
sed -i "s~\(train_filenames *: *\).*~\1$train_filenames~" temp_config.yml
sed -i "s~\(val_filenames *: *\).*~\1$val_filenames~" temp_config.yml
sed -i "s~\(test_filenames *: *\).*~\1$test_filenames~" temp_config.yml
sed -i "s~\(root_dir *: *\).*~\1$root_dir~" temp_config.yml
sed -i "s~\(log_dir *: *\).*~\1$LOG_DIR~" temp_config.yml

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_img_gen --config temp_config.yml
echo "job done"
