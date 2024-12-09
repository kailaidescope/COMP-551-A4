#!/bin/bash
# Access the job name
job_number="local-$1"

echo "BERT Test Job" > ../../output/win/job-$job_number/test_out.txt
py -3.11 ../../python/BERT_experiments/over_epochs.py ../../output/win/job-$job_number "head+1" 8 >> job.out 2>> job.err #insert python script here

