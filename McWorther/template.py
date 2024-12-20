import re

def prj_from_template(param, template, prjfile):
    param = add_prefix_suffix(param, r"$$")
    sed(param, template, prjfile)


def add_prefix_suffix(d, prefix_suffix):
    """
    Adds a prefix and suffix to the keys in the dictionary.

    Args:
        d (dict): Input dictionary
        prefix_suffix (str): The prefix and suffix to add to each key

    Returns:
        dict: New dictionary with modified keys
    """
    new_dict = {}
    for key, value in d.items():
        new_key = f"{prefix_suffix}{key}{prefix_suffix}"
        new_dict[new_key] = value
    return new_dict

def sed(replace, source, output):
    """Replaces strings in source file and writes the changes to the output file.

    In each line, replaces pattern with replace.

    Args:
        replace (dict)
            key (str): pattern to match (can be re.pattern)
            value (str): replacement str
        source  (str): input filename
        output  (str): output filename
    """

    with open(source, 'r') as fin:
        with open(output, 'w') as fout:
            num_replaced = 0

            for line in fin:
                out = line

                for k, v in replace.items():
                    new_out = re.sub(k, v, out)
                    out = new_out

                fout.write(out)

                if out != line:
                    num_replaced += 1

            print(f"{num_replaced} replacements made.")
