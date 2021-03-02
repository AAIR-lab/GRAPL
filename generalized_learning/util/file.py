
import json
import os
import pathlib
import re


def get_file_list(directory, file_extension_regex):

    extension_regex = re.compile(file_extension_regex)

    file_list = []
    for root, unused_dirname, filenames in os.walk(directory):
        for filename in filenames:
            if extension_regex.match(filename) is not None:

                file_list.append(pathlib.Path(root, filename))

    return file_list


def get_relative_path(name, parent_dir=None):

    path = pathlib.Path(name)
    path = path.expanduser()

    if path.is_absolute():

        return path
    else:

        assert parent_dir is not None
        path = parent_dir / path
        return path


def extract_contiguous(file_handle, regex, groups, first_line=None):

    def _get_regex_group(string, regex, groups):

        matched_groups = {}

        regex_match = regex.match(string)
        if regex_match is not None:

            for group in groups:

                matched_groups[group] = regex_match.group(group)

        return matched_groups

    entered = False
    regex_groups = []

    # Check if the first line is specified.
    if first_line is not None:

        matched_groups = _get_regex_group(
            first_line, regex, groups)
        if len(matched_groups) > 0:

            entered = True
            regex_groups += matched_groups

    # Iterate through the file handle.
    for line in file_handle:

        matched_groups = _get_regex_group(
            line, regex, groups)
        if len(matched_groups) > 0:

            entered = True
            regex_groups.append(matched_groups)
        elif entered:

            return regex_groups, line

    return regex_groups, None


# Changing the magic string can break compatibility with earlier stuff.
_PROPERTY_STR_MAGIC = "<magic_json>"
_PROPERTY_REGEX = re.compile("(\w|\W)* %s (?P<json>(\w|\W)*)($|\n)" % (
    _PROPERTY_STR_MAGIC))


def parse_properties(line):

    property_match = _PROPERTY_REGEX.match(line)
    if property_match is not None:

        json_str = property_match.group("json")
        json_str = json_str.strip()

        return json.loads(json_str)
    else:

        return None


def read_properties(file_path):

    file_handle = open(file_path, "r")

    property_dict = None
    for line in file_handle:

        line = line.strip()

        property_dict = parse_properties(line)
        if property_dict is not None:

            # Once a match is found, it is assumed that all properties are
            # written contiguously and there is only a single property block
            # per file.
            #
            # So, encountering a match implies that all properties for this
            # file have been parsed.
            break

    file_handle.close()
    return property_dict


def write_properties(file_handle, property_dict, prefix_str):

    json_str = json.dumps(property_dict)
    string = "%s %s %s\n\n" % (prefix_str, _PROPERTY_STR_MAGIC, json_str)
    file_handle.write(string)
