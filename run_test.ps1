# Activate virtual environment and run test
& ".\.venv\Scripts\Activate.ps1"
$env:CUDA_VISIBLE_DEVICES="0"
python main_task_offloading.py 2>&1 | Select-Object -First 150
