import csv
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument(
    "-acc",
    dest="acc_result",
    type=str,
    required=True,
)
parser.add_argument(
    "-prof",
    dest="prof_detail",
    type=str,
    required=True,
)
args = parser.parse_args(sys.argv[1:])
acc_result_path = args.acc_result
prof_detail_path = args.prof_detail
acc_summary = csv.reader(open(acc_result_path, 'r'))
prof_details = csv.reader(open(prof_detail_path, 'r'))

acc_dict = {}
acc_summary = list(acc_summary)
for i in range(len(acc_summary)):
    if i == 0:
        continue
    else:
        fwd_name = acc_summary[i][0] + ".forward"
        bwd_name = acc_summary[i][0] + ".backward"
        acc_dict[fwd_name] = acc_summary[i][1]
        acc_dict[bwd_name] = acc_summary[i][2]

prof_details = list(prof_details)
csv_head = tuple(prof_details[0]).__add__(tuple(["Acc_status"]))
for item in prof_details[1:]:
    name = item[0]
    try:
        status = acc_dict[name]
    except:
        status = "N/A"
    item.append(status)

with open('prof_summary.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_head)
    for item in prof_details[1:]:
        writer.writerow(item)