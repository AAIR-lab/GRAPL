
import argparse
import os
import pathlib
import shutil
import sys


def update_pythonpath():

    root = pathlib.Path(pathlib.Path(__file__).parent) / "../"

    sys.path.append(root.as_posix())
    sys.path.append((root / "generalized_learning").as_posix())

    fd_root_path = root / "dependencies" / "fast-downward_stochastic"
    sys.path.append(fd_root_path.as_posix())

    d2l_root_path = root / "dependencies" / "d2l"
    sltp_root_path = d2l_root_path / "src"

    sys.path.append(sltp_root_path.as_posix())

    tarski_root_path = root / "dependencies" / "tarski"
    tarski_src_path = tarski_root_path / "src"

    sys.path.append(tarski_src_path.as_posix())


update_pythonpath()


from generalized_learning.util import constants
from generalized_learning.util import rddl


def clean_rddl_file(directory, filename, keep_original=False):

    old_filepath = "%s/%s" % (directory, filename)

    is_domain = False
    new_filename = filename
    
    if rddl.is_domain_file(old_filepath):

        is_domain = True
        if not filename.endswith(constants.RDDL_DOMAIN_FILE_EXT):
            new_filename = filename.replace("rddl",
                                            constants.RDDL_DOMAIN_FILE_EXT)
    else:

        if not filename.endswith(constants.RDDL_PROBLEM_FILE_EXT):
            new_filename = filename.replace("rddl",
                                            constants.RDDL_PROBLEM_FILE_EXT)

    new_filepath = "%s/%s" % (directory, new_filename)

    if new_filepath != old_filepath:
        new_file_handle = open(new_filepath, "w")
        old_file_handle = open(old_filepath, "r")
    
        for line in old_file_handle:
    
            new_file_handle.write(line)
    
        new_file_handle.close()
        old_file_handle.close()

        if not keep_original:
    
            os.remove(old_filepath)

    if is_domain:

        pddl_filepath = rddl.write_rddl_domain_to_file(new_filepath, 
                                                       directory)
    else:

        pddl_filepath = rddl.write_rddl_problem_to_file(new_filepath, 
                                                        directory)

    return pathlib.Path(pddl_filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Convert RDDL files to PDDL files.")
    parser.add_argument("--input-dir", required=True, help="Base directory")
    parser.add_argument("--keep-original", action="store_true",
                        default=False,
                        help="Set this to keep the original files.")

    args = parser.parse_args()

    assert os.path.isdir(args.input_dir)

    for directory, _, filenames in shutil.os.walk(args.input_dir):

        for filename in filenames:

            if filename.endswith(".rddl") \
                and not (filename.endswith(constants.RDDL_PROBLEM_FILE_EXT)
                         or filename.endswith(constants.RDDL_DOMAIN_FILE_EXT)):

                try:
                    clean_rddl_file(directory, filename, args.keep_original)
                except Exception:

                    print("%s/%s failed to parse." % (directory, filename))
