#!/bin/bash
echo "Starting worker $1 on $(hostname)"
echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo "nvidia-smi failed, aborting"
    exit
fi

# Set root dirs
ROOT=/home/ishit/temp_root/
OUTPUT=/home/ishit/temp_out/

# get code
cd $ROOT

if [ ! -d $ROOT"/localImplicit_release" ]
then
    git clone https://github.com/ishit/localImplicit_release.git
else
    cd $ROOT"/localImplicit_release"
    git pull
    cd ..
fi

# get data
wget -nc http://132.239.95.124:8000/cifar.tar.gz 

# move code and data to shared memory
CODE=$ROOT
#CODE_LOCAL=/dev/shm/$1/modsiren/
CODE_LOCAL=/dev/shm/2/modsiren/
mkdir -p $CODE_LOCAL

echo "Copying code and data to local memory drive"
#mkdir -p /dev/shm/$1
mkdir -p /dev/shm/2
rsync -av --ignore-existing $CODE $CODE_LOCAL

# install python modules
cd $CODE_LOCAL"localImplicit_release"
echo "current dir" $(pwd)
echo "Installing python modules"
#export PYTHONPATH=.:$PYTHONPATH
pip install --user -r requirements.txt
pip install lpips # didn't worky locally with requirements.txt!

# get config scripts
CONFIG_DIR=$CODE_LOCAL"localImplicit_release/config/img_gen/celeba"

echo $CONFIG_DIR
configs=()
for c in "$CONFIG_DIR"/*.yml
    do
        configs+=($c)
    done

let jobid=$1+1
echo "Running job" $jobid on worker $1
curr_config="${configs[$jobid]}"

# extract data
cd $CODE_LOCAL
echo extracting data
if [ ! -d "cifar" ] 
then
    tar zvxf cifar.tar.gz
fi

echo config: $curr_config
echo Training
mkdir -p $OUTPUT

# edit config file
cd $CODE_LOCAL"localImplicit_release"
cp $curr_config temp_config.yml

#LOG_DIR=$OUTPUT"/logs_cifar"
LOG_DIR="/home/ishit/Documents/localImplicit_release/data_logs/ishit/logs_cifar/"
train_filenames=$CODE_LOCAL"/cifar/train_filenames.txt"
val_filenames=$CODE_LOCAL"/cifar/val_filenames.txt"
test_filenames=$CODE_LOCAL"/cifar/val_filenames.txt"
root_dir=$CODE_LOCAL"/cifar/"

echo $val_filenames
sed -i "s~\(train_filenames *: *\).*~\1$train_filenames~" temp_config.yml
sed -i "s~\(val_filenames *: *\).*~\1$val_filenames~" temp_config.yml
sed -i "s~\(test_filenames *: *\).*~\1$test_filenames~" temp_config.yml
sed -i "s~\(root_dir *: *\).*~\1$root_dir~" temp_config.yml
sed -i "s~\(log_dir *: *\).*~\1$LOG_DIR~" temp_config.yml
sed -i "s~\(crop_size *: *\).*~\1 32~" temp_config.yml
sed -i "s~\(celeba\)~cifar~" temp_config.yml
sed -i "s~\(mode *: *\).*~\1test~" temp_config.yml

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_img_gen --config temp_config.yml
echo "job done"
