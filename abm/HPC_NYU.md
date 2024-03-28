# connect to the cluster

ssh jc12343@greene.hpc.nyu.edu

## move to scratch dir and clone repo

cd $SCRATCH
git clone https://github.com/ChaosDonkey06/AMR_ABM2.git

## copy the data movement data from DB to the cluster
mkdir shaman_lab
mv AMR_ABM2 ./shaman_lab
mkdir shaman_lab/amr-hospitals
mkdir shaman_lab/amr-hospitals/data
mkdir shaman_lab/amr-hospitals/data/long_files_8_25_2021

scp /Users/chaosdonkey06/Dropbox/shaman-lab/amr-hospitals/data/long_files_8_25_2021/patient_movement_2022-Nov.csv jc12343@greene.hpc.nyu.edu:/scratch/jc12343/shaman_lab/amr-hospitals/data/long_files_8_25_2021/

scp /Users/chaosdonkey06/Dropbox/shaman-lab/amr-hospitals/data/long_files_8_25_2021/amro_ward.csv jc12343@greene.hpc.nyu.edu:/scratch/jc12343/shaman_lab/amr-hospitals/data/long_files_8_25_2021/

scp /Users/chaosdonkey06/Desktop/Shaman-lab/AMR_ABM2/global_config/config_file.csv jc12343@greene.hpc.nyu.edu:/scratch/jc12343/shaman_lab/AMR_ABM2/global_config/

## clone and install the inference package
cd /scratch/jc12343/shaman_lab/AMR_ABM2
git clone -b numpy-version https://github.com/ChaosDonkey06/pompjax.git
cd pompjax
pip install .


## submit the jobs for each amro (indexed)
sbatch inference_readmission_amro.sh 0
sbatch inference_readmission_amro.sh 1
sbatch inference_readmission_amro.sh 2
sbatch inference_readmission_amro.sh 3
sbatch inference_readmission_amro.sh 4
sbatch inference_readmission_amro.sh 5
sbatch inference_readmission_amro.sh 6

