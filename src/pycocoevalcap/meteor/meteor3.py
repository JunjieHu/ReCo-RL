import subprocess, threading

class MeteorScorer:
    def __init__(self, jar_path, lang=None, paraphrase_dir=None):
        if lang in ['en', 'cz', 'de', 'es', 'fr', 'ar']:
            self.cmd = "java -Xmx2G -jar " + jar_path + " - - -l " + lang + " -stdio"
        elif paraphrase_dir is not None:
            self.cmd = "java -Xmx2G -jar " + jar_path + " - - -new " + paraphrase_dir + " -stdio"
        else:
            print('either lang and paraphrase_dir should be provided')
            exit(0)

        self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines = True, bufsize = 1, shell=True)
        self.lock = threading.Lock()

    def score(self, hyp, ref):
        self.lock.acquire()
        self.process.stdin.write("SCORE ||| {} ||| {} \n".format(ref, hyp))
        stdout = self.process.stdout.readline()
        self.process.stdin.write("EVAL ||| {}".format(stdout))
        stdout = self.process.stdout.readline()
        score = float(stdout)
        self.lock.release()
        return score

def read_file(file):
    sent = []
    for line in open(file):
        sent.append(line.strip())
    return sent

def score(hyp_file, ref_file):
    hyps = read_file(hyp_file)
    refs = read_file(ref_file)

    dir = '/home/t-juhu/work/seq2seq/pycocoevalcap/meteor'
    meteor = MeteorScorer(jar_path='/home/t-juhu/work/seq2seq/pycocoevalcap/meteor/meteor-*.jar', paraphrase_dir=dir, lang='en')
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        print('Score', meteor.score(hyp, ref))


# Example
# hyp_file = '/usr2/data/junjieh/Research/questplusplus/lang_resources/japanese/AlGore_2009_ja_clean_tok-hyp.txt'
# ref_file = '/usr2/data/junjieh/Research/questplusplus/lang_resources/japanese/AlGore_2009_ja_clean_tok-ref.txt'
# score(hyp_file, ref_file)
score('test.txt', 'ref.txt')