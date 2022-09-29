from shutil import copyfile
from pathlib import Path
files = sorted(Path('.').glob('*.json'))
parent = Path('..','sample_small_dataset')
for f in files[:200]:
    copyfile(f, Path(parent, 'train',f.name))
for f in files[200:250]:
    copyfile(f, Path(parent, 'val',f.name))
for f in files[250:300]:
    copyfile(f, Path(parent, 'test', f.name))
