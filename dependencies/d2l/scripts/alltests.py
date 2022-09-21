import logging
import os
import subprocess
import sys
import tempfile

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

tests = [
    ("gripper:aaai_prob01_gc",
     ["total complexity: 8", "Num[free]", "Num[Not(Equal(at_g,at))]", "Num[Exists(carry,<universe>)]",
      "Num[Exists(at_g,at-robby)]"]),

    ("blocks:aaai_clear_x_simple_hybrid",
     ["total complexity: 8",
      # "Bool[holding]", (sometimes  Atom[handempty] is found instead)
      "Bool[And(holding,Nominal(a))]",
      "Num[Exists(Star(on),Nominal(a))]"]),

    ("blocks:aaai_clear_x_simple_hybrid_gc",
     ["total complexity: 8",
      # "Bool[holding]", (sometimes  Atom[handempty] is found instead)
      "Bool[And(clear_g,holding)]",
      "Num[Exists(Star(on),clear_g)]"]),

    ("blocks:aaai_bw_on_x_y_completeness_opt",
     ["total complexity: 17",
      # "Bool[holding]",
      "Bool[And(holding,Nominal(a))]",
      "Num[Exists(Star(on),Nominal(b))]", "Num[Exists(Star(on),Nominal(a))]",
      "Bool[And(Exists(on,Nominal(b)),Nominal(a))]"]),
]


def test(script, configuration, expected_output):
    cwd = os.path.join(BASEDIR, "experiments")
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as f:
        command = [os.path.join(cwd, script), configuration]
        print('Calling "{}". Output redirected to "{}"'.format(' '.join(command), f.name))
        retcode = subprocess.call(command, stdout=f, stderr=f)
        if retcode:
            print("Experiment returned error code {}".format(retcode))
            sys.exit(-1)

        f.flush()
        f.seek(0)
        output = f.read()
        # print(output)
        for s in expected_output:
            if s not in output:
                print('Expected string "{}" not found in output. Check experiment output at "{}"'.format(s, f.name))
                sys.exit(-1)


def runtests():
    for configuration, expected_output in tests:
        test("run.py", configuration, expected_output)
    print("All tests OK")


if __name__ == "__main__":
    runtests()
