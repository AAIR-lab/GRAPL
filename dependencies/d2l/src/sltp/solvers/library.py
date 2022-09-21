

class WPM3:
    """ See http://web.udl.es/usuaris/q4374304/ """
    TAG = 'wpm3-co'

    def __init__(self, rundir=None):
        self.run_dir = rundir

    @staticmethod
    def command(input_filename):
        return ['WPM3-2015-co', input_filename]


class Maxino:
    TAG = 'maxino'

    def __init__(self, rundir=None):
        self.run_dir = rundir

    @staticmethod
    def command(input_filename):
        return ['maxino', input_filename]


class Openwbo:
    TAG = 'openwbo'

    def __init__(self, rundir=None):
        self.run_dir = rundir

    @staticmethod
    def command(input_filename):
        return ['/home/local/anonymous/anonymous/work/git/d2l/open-wbo_static', input_filename]


class OpenwboInc:
    TAG = 'openwbo-inc'

    def __init__(self, rundir=None):
        self.run_dir = rundir

    @staticmethod
    def command(input_filename):
        # From competition script: open-wbo-inc -ca=1 -c=100000 -algorithm=6
        return ['open-wbo-inc', '-ca=1', '-c=100000', '-algorithm=6', input_filename]


class Glucose:
    TAG = 'glucose'

    def __init__(self):
        pass

    @staticmethod
    def command(input_filename):
        return ['glucose_static', input_filename]


class GlucoseSyrup:
    TAG = 'glucose-syrup'

    def __init__(self):
        pass

    @staticmethod
    def command(input_filename):
        return ['glucose-syrup_static', input_filename]
