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
        csv_content = ""
        rows = f.readlines()
        for rIdx, row in enumerate(rows):
            cols = row.split(',')
            event = cols[-1]
            color = "0"
            for n, c in event_map.items():
                if n in event:
                    color = c
            number_devs = len(cols) - 1
            dev_string = ",".join(
                [str(toMiB(int(cols[d]))) for d in range(0, number_devs)])
            if rIdx == 0:
                dev_head = ",".join(
                    ["dev" + str(d) for d in range(0, number_devs)])
                csv_content += "x," + dev_head + ",color\n"
            csv_content += str(rIdx) + "," + dev_string + "," + color + "\n"
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
