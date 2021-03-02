
# https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python


def round_no(x, prec=0, base=1):

    return round(base * round(float(x) / base), prec)
