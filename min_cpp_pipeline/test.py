from multiprocessing.sharedctypes import Value
# from subprocess import Popen, PIPE
import subprocess

from typing import Dict, Any
import json


def params_to_json(params: Dict[str, Any], output_path: str) -> None:
    """Method to write parameters to a json file.

    Args:
        params: python dictionary of parameters
        output_path: path to save json file
    """
    with open(output_path, "w") as json_file:
        json.dump(params, json_file)


def params_to_txt(params: Dict[str, Any], output_path: str) -> None:
    """Method to write parameters to a .txt file.

    Args:
        params: python dictionary of parameters
        output_path: path to save txt file
    """
    with open(output_path, "w") as txt_file:
        for i, (k, v) in enumerate(params.items()):
            if isinstance(v, int):
                value_type = "int"
            elif isinstance(v, str):
                value_type = "str"
            elif isinstance(v, float):
                value_type = "float"
            elif isinstance(v, bool):
                v = int(v)
                value_type = "int"
            else:
                raise ValueError(f"type {type(v)} not yet handled in params_to_txt.")
            if i == len(params) - 1:
                line_end = ""
            else:
                line_end = "\n"
            txt_file.write(f"{value_type};{k};{v}{line_end}")


# process = Popen(["./a.out"], shell=True, stdout=PIPE, stdin=PIPE)

params = {"N": 1000, "lr": 0.1}

params_to_txt(params, "test.txt")

cmd = "./a.out"
print(cmd.split())
args = "test.txt output.txt".split()
print(cmd.split() + args)
subprocess.call(cmd.split() + args, shell=False)

# process.stdin.write(bytes("test.txt", 'UTF-8'))
# process.stdin.flush()

# result = process.stdout.readline().strip()
# import pdb

# pdb.set_trace()
# print(result)
