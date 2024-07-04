#!/bin/bash
echo "Starting worker $1 on $(hostname)"
echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo "nvidia-smi failed, aborting"
    echo $(hostname)
    exit
fi

# Set root dirs
DATA=/newfoundland2/ishit/div2k/
ROOT=/home/ishit/temp_root/
OUTPUT=/home/ishit/temp_out/

# # get code
 cd $ROOT

 if [ ! -d $ROOT"/localImplicit_release" ]
 then
     git clone https://github.com/ishit/localImplicit_release.git
 else
     cd $ROOT"/localImplicit_release"
     git pull
     cd ..
 fi

# move code and data to shared memory
CODE=$ROOT
CODE_LOCAL=/dev/shm/ishit/$1
mkdir -p $CODE_LOCAL

echo "Copying code and data to local memory drive"
echo $CODE $(dirname $CODE_LOCAL)
#mkdir -p /dev/shm/$1
rsync -av --ignore-existing $CODE $(dirname $CODE_LOCAL)

# install python modules
cd $CODE_LOCAL
conda activate sdf_exp
echo "current dir" $(pwd)
echo "Installing python modules"
export PYTHONPATH=.:$PYTHONPATH
pip install -r requirements.txt
pip install lpips # didn't worky locally with requirements.txt!

# get config scripts
CONFIG_DIR=$CODE_LOCAL/config/single_image/

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
rsync -av --ignore-existing $DATA/DIV2K_train_HR.zip .

echo extracting data
if [ ! -d "DIV2K_train_HR" ] 
then
    unzip DIV2K_train_HR.zip
fi

echo config: $curr_config
echo Training
mkdir -p $OUTPUT

# edit config file
cd $CODE_LOCAL
cp $curr_config temp_config.yml

#LOG_DIR=$OUTPUT/logs_div2k

#echo $val_filenames
#sed -i "s~\(log_dir *: *\).*~\1$LOG_DIR~" temp_config.yml

#for i in $(seq -f "$04g" 1 50)
#do
    #echo $1
    #EXP_NAME=$OUTPUT/logs_div2k
#done

#CUDA_VISIBLE_DEVICES=0 python -m scripts.train_img_gen --config temp_config.yml
#echo "job done"

