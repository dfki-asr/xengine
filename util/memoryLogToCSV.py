#!/usr/bin/env python3

import argparse


def toMiB(memory_in_bytes):
    return float(memory_in_bytes / 1024.0 / 1024.0)


def readStringFile(txtfile, csvfile):
    event_map = {
        "init_program": "0",
        "loaded_params": "1",
        "fwd": "2",
        "bwd": "3",
        "finished_run": "4"
    }
    with open(txtfile) as f:
        csv_content = "x,cpu,gpu,color\n"
        rows = f.readlines()
        for rIdx, row in enumerate(rows):
            cols = row.split(',')
            event = cols[2]
            color = "0"
            for n, c in event_map.items():
                if n in event:
                    color = c
            csv_content += str(rIdx) + "," + str(toMiB(int(
                cols[0]))) + "," + str(toMiB(int(
                    cols[1]))) + "," + color + "\n"
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
