#!/usr/bin/env python3

import argparse


def readStringFile(txtfile, csvfile):
    with open(txtfile) as f:
        csv_content = "x,y\n"
        rows = f.readlines()
        startMemory = 0
        for rIdx, row in enumerate(rows):
            if rIdx == 0:
                startMemory = int(row)
                print("Subtract initial memory ", startMemory,
                      " MiB from all Y values.")
            csv_content += str(rIdx) + "," + str(int(row) - startMemory) + "\n"
        print(csv_content)
        file = open(csvfile, 'w')
        file.write(csv_content)
        file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert memory log files to csv")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="The input memory log file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The output csv file.",
    )
    args = parser.parse_args()
    readStringFile(args.input, args.output)


if __name__ == "__main__":
    main()
