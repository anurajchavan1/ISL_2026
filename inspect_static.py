import zipfile
import json

with zipfile.ZipFile('isl_landmark_model.keras', 'r') as z:
    with z.open('config.json') as f:
        config = json.load(f)
        print(json.dumps(config, indent=2))