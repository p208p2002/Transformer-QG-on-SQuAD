import os
if __name__ == "__main__":
    # nlg-eval for `Our Scorer`
    os.system('sudo apt-get -y update && sudo apt-get -y install default-jre && sudo apt-get -y install default-jdk')
    os.system('export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup')
    import stanza
    stanza.download('en')

    # NQG Scorer
    # original nqg scorer run with py2, we maintain a py3 version
    # see this PR for more info: https://github.com/xinyadu/nqg/pull/44
    os.system('git clone -b python3 https://github.com/p208p2002/nqg.git')