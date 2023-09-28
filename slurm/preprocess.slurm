#!/bin/bash

#SBATCH --job-name prep-virtifier
#SBATCH --output logs/preprocess.log
#SBATCH --error logs/preprocess.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
#SBATCH --cpus-per-task 1
#SBATCH --partition allgroups
#SBATCH --mem 500G

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
conda activate virtifier

cd ..

echo "*** Preprocessing training datasets"
echo "** Preprocess hosts..."
srun python preprocessing.py --infile ../DeepVirFinder/datasets/training/host-training.fa --outfile host-training.csv
echo "** Preprocess virus..."
srun python preprocessing.py --infile ../DeepVirFinder/datasets/training/virus-training.fa --outfile virus-training.csv

echo "*** Preprocessing test datasets"
echo "** Preprocess hosts..."
srun python preprocessing.py --infile ../DeepVirFinder/datasets/test/host-test.fa --outfile host-test.csv
echo "** Preprocess virus..."
srun python preprocessing.py --infile ../DeepVirFinder/datasets/test/virus-test.fa --outfile virus-test.csv

echo "*** Merging training datasets"
cat virus-training.csv host-training.csv > data-training.csv
echo "** Number of virus lines: $(wc -l virus-training.csv)"
echo "** Number of host lines: $(wc -l host-training.csv)"
rm virus-training.csv host-training.csv

echo "*** Merging test datasets"
cat virus-test.csv host-test.csv > data-test.csv
echo "** Number of virus lines: $(wc -l virus-test.csv)"
echo "** Number of host lines: $(wc -l host-test.csv)"
rm virus-test.csv host-test.csv

echo "All done!"
