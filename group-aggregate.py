#!/usr/bin/python3
"""Group and aggregate record fields"""

import sys
import argparse
import builtins
import operator
import itertools
import functools


AGGREGATORS = ('count', 'sum', 'avg')


def count(iterable):
	return builtins.sum(1 for _ in iterable)


def sum(iterable):
	return builtins.sum(map(float, iterable))


def avg(iterable):
	try:
		count = len(iterable)
	except TypeError:
		total, count = functools.reduce(
			lambda acc, x: (acc[0] + x, acc[1] + 1), map(float, iterable), (0, 0))
	else:
		total = sum(iterable)
	return total / (count or float('nan'))


def parse_aggregator(s):
	s = s.split(':', 2)
	s[0] = int(s[0])
	s[1] = AGGREGATORS[s[1]]
	if len(s) == 2: s.append('')
	return s


def parse_args():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument('groups',
		type=lambda s: tuple(map(int, s.split(','))),
		help='A list of field indexes used to group records (zero-based, '
			'comma-separated).')
	ap.add_argument('aggregators',
		nargs='+', type=parse_aggregator,
		help='A field index (zero-based), the name of an aggregation function '
			'({:s}), and optionally a format string, all colon-separated.'
			.format(', '.join(AGGREGATORS)))
	ap.add_argument('-F', '--input-field-separator', metavar='SEP',
		help='The input field separator string. (default: a series of white-space '
			'characters)')
	ap.add_argument('-O', '--output-field-separator', metavar='SEP',
		default='\t',
		help='The output field separator string. (default: tab character)')
	args = ap.parse_args()

	err = tuple(filter((0).__gt__, itertools.chain(
		args.groups, map(operator.itemgetter(0), args.aggregators))))
	if err:
		ap.error('Invalid field indices: ' + ', '.join(map(str, err)))

	err = tuple(filter(args.groups.__contains__,
		map(operator.itemgetter(0), args.aggregators)))
	if err:
		ap.error('Cannot aggregate grouping fields: ' + ', '.join(map(str, err)))

	return args


def main():
	args = parse_args()
	records = map(
		functools.partial(str.split, sep=args.input_field_separator or None),
		sys.stdin)
	print = functools.partial(builtins.print, sep=args.output_field_separator)

	if not args.groups:
		groups = ((), records)
	else:
		if len(args.groups) == 1:
			def group_key_func(record, idx=args.groups[0]):
				return (record[idx],)
		else:
			group_key_func = operator.itemgetter(*args.groups)
		groups = itertools.groupby(records, group_key_func)

	for group_key, group_records in groups:
		aggregated_fields = (
			format(
				aggregator(map(operator.itemgetter(idx), group_records)),
				formatter)
			for idx, aggregator, formatter in args.aggregators)
		print(*itertools.chain(group_key, aggregated_fields))


AGGREGATORS = dict(zip(AGGREGATORS, map(globals().__getitem__, AGGREGATORS)))

if __name__ == '__main__':
	main()
