import os

raw = 'raw_videos'
total_videos = 0
classes = []

for root, dirs, files in os.walk(raw):
    videos = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    if videos:
        class_name = os.path.basename(root)
        classes.append((class_name, len(videos)))
        total_videos += len(videos)

classes.sort()

print(f'Total classes : {len(classes)}')
print(f'Total videos  : {total_videos}')
print()
print(f'{"Class":<30} {"Videos":>6}')
print('-' * 38)
for name, count in classes:
    print(f'{name:<30} {count:>6}')