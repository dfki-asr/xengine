#!/usr/bin/env python3

import os
import argparse


def readStringFile(txtfile, csvfile):
    with open(txtfile) as f:
        csv_content = ""
        rows = f.readlines()
        print("Y range: 0, ..., " + str(-len(rows) + 1))
        for rIdx, row in enumerate(rows):
            if rIdx == 0 or rIdx == len(rows) - 1:
                pass
            for cIdx, col in enumerate(row):
                if cIdx == 0 or cIdx == len(row) - 1:
                    pass
                if col == " ":
                    pass
                elif col == "\n":
                    pass
                elif col == "\u2588":
                    csv_content += str(cIdx) + "," + str(
                        -rIdx) + "," + "1" + "\n"
                else:
                    pass
        file = open(csvfile, 'w')
        file.write(csv_content)
        file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert txt schedules to csv")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="The input txt file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The output csv file.",
    )
    args = parser.parse_args()
    path_in = args.input
    path_out = args.output

    if os.path.isfile(path_in) and os.path.isfile(path_out):
        readStringFile(path_in, path_out)

    if os.path.isdir(path_in) and not os.path.isfile(path_out):
        os.makedirs(path_out, exist_ok=True)
        listed_files = os.listdir(path_in)
        for f in listed_files:
            f_in = "/".join([path_in, f])
            f_out = "/".join([path_out, f.replace("txt", "csv")])
            print("in: ", f_in)
            print("out: ", f_out)
            readStringFile(f_in, f_out)


if __name__ == "__main__":
    main()
