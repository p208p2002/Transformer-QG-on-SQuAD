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

# drcd
os.makedirs('datasets/drcd',exist_ok=True)
os.system('wget -O datasets/drcd/train.json https://github.com/DRCKnowledgeTeam/DRCD/raw/master/DRCD_training.json')
os.system('wget -O datasets/drcd/test.json https://github.com/DRCKnowledgeTeam/DRCD/raw/master/DRCD_test.json')
os.system('wget -O datasets/drcd/dev.json https://github.com/DRCKnowledgeTeam/DRCD/raw/master/DRCD_dev.json')


