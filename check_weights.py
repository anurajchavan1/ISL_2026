import zipfile
import os

print("Files inside isl_landmark_model.keras:")
with zipfile.ZipFile('isl_landmark_model.keras', 'r') as z:
    for name in z.namelist():
        print(f"  {name}")

print("\nChecking extracted folder:")
for root, dirs, files in os.walk('_keras_extracted'):
    for f in files:
        path = os.path.join(root, f)
        print(f"  {path}")