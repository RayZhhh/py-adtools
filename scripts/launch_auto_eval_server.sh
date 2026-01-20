cd ../
python -m adtools.evaluator.auto_server\
    -d usage/example_evaluator.py\
    --host 0.0.0.0\
    --port 8000\
    -t 10\
    --max-workers 4