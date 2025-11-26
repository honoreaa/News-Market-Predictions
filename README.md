# Deployment Guide

1. After forking or loading into the repository --> Install requirements

       `pip install -r requirements.txt`

2. Enter News-Market-Predictions directory

3. Check if embeddings exist on your device

       `ls -lh data/embeddings/ 2>/dev/null || echo "Run embeddings.py :)"`

  3.a Create embeddings if they don't exist for some reason (may take 10-20 minutes depending on your device)

     `python3 embedding.py`

4. Run train.py

       `python3 -m ML.train --config config.yaml`

Should be good! Please reach out to haalexander@ucdavis.edu if you have any issues. 

