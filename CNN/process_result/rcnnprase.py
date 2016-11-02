#!/usr/bin/env python
import os
import re
import argparse
import csv
from collections import OrderedDict

#create a list that handle the outcome

def parse_log(path_to_log):
	regex_batch = re.compile('Batch (\d+), (\S+) = ([\.\deE+-]+)')
	output_num = -1
	test_dict_list = []
	test_row = None
	print "wocao"

	with open(path_to_log) as f:
		for line in f:
			matcher = regex_batch.search(line)
			if matcher:
				output_num = matcher.group(1)
			if output_num	 == -1:
				continue

			test_dict_list , test_row = parse_line_for_net(regex_batch,test_row \
				, test_dict_list,line,output_num)
	return test_dict_list

def parse_line_for_net(regex_obj,row,row_dict_list,line,num):
	output_match = regex_obj.search(line)
	if output_match:
		if not row or row['Num'] != num:
			if row:
				row_dict_list.append(row)

			row = OrderedDict([('Num',num)])

		name = output_match.group(2)
		value = output_match.group(3)
		row[name] = float(value)

	if row and len(row_dict_list) >=1 and len(row) == len(row_dict_list[0]):
		row_dict_list.append(row)
		row = None

	return row_dict_list,row



def save_csv_files(logfile_path,output_dir,test_dict_list,delimiter=',',verbose = False):
	log_basename = os.path.basename(logfile_path)
	test_filename = os.path.join(output_dir, log_basename)
	write_csv(test_filename, test_dict_list, delimiter, verbose)

def write_csv(output_filename, dict_list, delimiter, verbose=False):
	dialect = csv.excel
	dialect.delimiter = delimiter

	with open(output_filename, 'w') as f:
		dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
		dict_writer.writeheader()
		dict_writer.writerows(dict_list)
	if verbose:
		print 'Wrote %s' % output_filename

def parse_args():
	description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
	parser = argparse.ArgumentParser(description=description)

	parser.add_argument('logfile_path',
                        help='Path to log file')

	parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

	parser.add_argument('--verbose',action='store_true',
                        help='Print some extra info (e.g., output filenames)')

	parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	test_dict_list = parse_log(args.logfile_path)
	save_csv_files(args.logfile_path, args.output_dir,test_dict_list, delimiter=args.delimiter)

if __name__ == '__main__':
	print"hello becky"
	main()