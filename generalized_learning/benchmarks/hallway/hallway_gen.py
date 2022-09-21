import argparse


TOTAL_LOCATIONS = 5

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", required=True, type=int)
    parser.add_argument("-d", required=True)

    args = parser.parse_args()

    TOTAL_LOCATIONS = args.n

    file_handle = open("%s/problem_0.problem.pddl" % (args.d), "w")

    file_handle.write("(define (problem hallway-%u)\n" % (TOTAL_LOCATIONS))
    file_handle.write("(:domain hallway)\n")

    file_handle.write("(:objects")

    for i in range(TOTAL_LOCATIONS):

        file_handle.write(" n%u" % (i))

    file_handle.write(" - node)\n")

    file_handle.write("(:init")

    s = 0
    t = 1
    total_edges = (TOTAL_LOCATIONS - 1) * 2

    edge = 0
    while edge < total_edges:

        file_handle.write(" (adjacent n%u n%u)\n" % (s, t))
        file_handle.write(" (adjacent n%u n%u)\n" % (t, s))

        s += 1
        t += 1

        edge += 2

    file_handle.write(" (at n0)\n")

    file_handle.write(")\n")

    file_handle.write("(:goal (and")

    file_handle.write(" (painted n%u)\n" % (TOTAL_LOCATIONS - 1))

    file_handle.write("))\n")
    file_handle.write(")\n")

