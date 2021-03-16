import os

os.system('rm -rf datasets/')

# squad

# squad 73k https://github.com/xinyadu/nqg
os.makedirs('datasets/squad-nqg',exist_ok=True)
os.system('wget -P datasets/squad-nqg https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/dev.json')
os.system('wget -P datasets/squad-nqg https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/test.json')
os.system('wget -P datasets/squad-nqg https://raw.githubusercontent.com/xinyadu/nqg/master/data/raw/train.json')

# squad 81k
os.makedirs('datasets/squad_81k',exist_ok=True)