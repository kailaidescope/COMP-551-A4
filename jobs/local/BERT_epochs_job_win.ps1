# Access the job name
Param (
    [string]$JobNumber
)

$job_number = "local-$JobNumber"
$job_path = "../../output/win/job-$job_number"

# Create the directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $job_path | Out-Null

# Create the test_out.txt file with the specified content
"BERT Test Job $job_number" | Set-Content -Path "$job_path/test_out.txt"

# Run the Python script and redirect stdout and stderr to respective files
python "../../python/BERT_experiments/over_epochs.py" `
    "$job_path" `
    "head+1" `
    8 `
    $true `
    >> "$job_path/job.out" `
    2>> "$job_path/job.err"
