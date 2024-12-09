
# Access the job name
job_number="local-$1"
job_path="../../output/mac/job-$job_number"

mkdir -p $job_path
echo "BERT Test Job $job_number" > $job_path/test_out.txt
python "../../python/BERT_experiments/over_epochs.py" "$job_path" "head+1" 8 True >> $job_path/job.out 2>> $job_path/job.err #insert python script here

