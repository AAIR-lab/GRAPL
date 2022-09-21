import logging
import shutil
import time

from ..version import get_version


def get_terminal_size():
    return shutil.get_terminal_size()


def header(text, indent=0):
    term_s = shutil.get_terminal_size()
    n_free = min(max((term_s.columns - indent*8), 0), 80)
    sep = "="*n_free
    return lines([sep, text, sep], indent)


def lines(text, indent=0):
    text = [text] if isinstance(text, str) else text
    idt = "\t"*indent
    return idt + ("\n"+idt).join(text)


def log_time(f, lvl=logging.INFO, msg=None):
    logging.log(lvl, msg)
    t0 = time.process_time()
    result = f()
    logging.log(lvl, "\t-> Elapsed CPU time: {0:.2f}".format(time.process_time() - t0))
    return result


def print_hello():
    print(header("SLTP v.{}".format(get_version())))
