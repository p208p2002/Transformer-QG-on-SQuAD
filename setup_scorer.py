import os
if __name__ == "__main__":
    os.system('sudo apt-get -y update && sudo apt-get -y install default-jre && sudo apt-get -y install default-jdk')
    os.system('sudo pip install git+https://github.com/voidful/nlg-eval.git@master')
    os.system('pip install git+https://github.com/Tiiiger/bert_score')
    os.system('export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup')