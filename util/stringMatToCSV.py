#!/usr/bin/env python3

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
    readStringFile(args.input, args.output)


if __name__ == "__main__":
    main()
