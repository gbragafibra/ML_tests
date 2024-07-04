#!/usr/bin/env python
import os
from argparse import ArgumentParser, Namespace
from typing import List

def arguments() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument(
		"-d",
		"--data_dir",
		help="Dir containing sub_dirs for each molecule",
		type=str,
		required=True)
	parser.add_argument(
		"-o",
		"--output_file",
		help="Output file containing the keys",
		type=str,
		required=True)
	args = parser.parse_args()
	return args

def write_keys(keys: List[str], file: str) -> None:
	with open(file, "w") as w:
		for key in keys:
			w.write(key + "\n")
	return


def main(args: Namespace) -> None:
	keys = []
	for root, dirs, _ in os.walk(args.data_dir):
		for dir_name in dirs:
			keys.append(dir_name)
		break  

	write_keys(keys, args.output_file)
	return

if __name__ == "__main__":
	args = arguments()
	main(args)