#!/usr/bin/python3
"""Group and aggregate record fields"""

import re
import sys
import math
import array
import codecs
import argparse
import itertools
from operator import itemgetter, attrgetter, methodcaller
from functools import reduce, partial as fpartial


try:
	RegexType = re._pattern_type
except AttributeError:
	RegexType = type()
else:
	assert isinstance(re.compile(''), RegexType)


def first(iterable):
	return next(iter(iterable), None)


def count(iterable):
	try:
		return len(iterable)
	except TypeError:
		return sum(1 for _ in iterable)


def isum(iterable):
	return sum(map(int, iterable))

def fsum(iterable):
	return math.fsum(map(float, iterable))


def iavg(iterable):
	try:
		count = len(iterable)
	except TypeError:
		total, count = reduce(
			lambda acc, x: (acc[0] + x, acc[1] + 1), map(int, iterable), (0, 0))
	else:
		total = isum(iterable)
	return total / (count or float('nan'))

def favg(iterable):
	try:
		count = len(iterable)
	except TypeError:
		iterable = array.array('d', map(float, iterable))
		count = len(iterable)
		total = math.fsum(iterable)
	else:
		total = fsum(iterable)
	return total / (count or float('nan'))


class Aggregation:

	__slots__ = ('field_index', 'aggregator', 'format')

	defaults = {'format': ''}

	aggregator_functions = ('first', 'count', 'isum', 'fsum', 'iavg', 'favg')
	aggregator_functions = dict(zip(
		aggregator_functions, map(globals().__getitem__, aggregator_functions)))


	def __init__(self, *args, **kwargs):
		if args and not kwargs:
			self._init_positional_args(args)
		elif kwargs and not args:
			self._init_kw_args(kwargs)
		else:
			raise TypeError('Must supply either positional or key-word arguments')


	def _init_positional_args(self, args):
		l_args = len(args)
		slots = self.__slots__
		l_slots = len(slots)
		defaults = self.defaults
		l_mandatory = l_slots - len(defaults)

		if l_args == 1:
			args = args[0].split(':', 2)
			l_args = len(args)
			if l_args < l_mandatory:
				raise ValueError('Invalid aggregation: ' + repr(':'.join(args)))
			self.field_index = parse_field_index(args[0])
			self.aggregator = self.aggregator_functions.get(args[1])
			if self.aggregator is None:
				raise ValueError('Invalid aggregation function name: ' + repr(args[1]))
			for k, v in zip(self.__slots__[2:], args[2:]):
				setattr(self, k, v)

		else:
			if l_args < l_mandatory:
				raise TypeError(
					'Too few arguments: {:d} < {:d}'.format(l_args, l_mandatory))
			if l_args > l_slots:
				raise TypeError(
					'Too many arguments: {:d} > {:d}'.format(l_args, l_slots))
			for k, v in zip(self.__slots__, args):
				setattr(self, k, v)

		for k in self.__slots__[l_args:]:
			setattr(self, k, defaults[k])


	def _init_kw_args(self, kwargs):
		print(*itertools.starmap('{:s}={!r}'.format, kwargs.items()), sep=', ')

		for k in self.__slots__:
			v = kwargs.pop(k, self)
			if v is self:
				v = self.defaults.get(k)
				if v is None:
					raise TypeError('Missing key-word argument: ' + k)
			setattr(self, k, v)

		if kwargs:
			raise TypeError('Surplus key-word arguments: ' + ', '.join(kwargs))


	def __repr__(self):
		return '{:s}({:s})'.format(
			type(self).__qualname__,
			', '.join(map('{:s}={!r}'.format,
				self.__slots__, map(fpartial(getattr, self), self.__slots__))))


def parse_field_index(s):
	start, sep, stop = s.partition('-')
	if not sep:
		return int(start)
	return slice(int(start) if start else 0, int(stop) if stop else None)


def alt_if_none(x, alt):
	return x if x is not None else alt


def slice_stop_alt(r, alt=float('inf')):
	return alt_if_none(r.stop, alt)


def parse_regex_flags(s):
	flags = 0
	try:
		for c in s:
			flags |= getattr(re, c.uppercase())
	except (AttributeError, TypeError):
		raise ValueError('Invalid regular expression flag: ' + repr(c))
	return flags


def validate_field_index(idx):
	if isinstance(idx, int):
		return idx > 0

	assert isinstance(idx, slice) and alt_if_none(idx.step, 1) == 1
	return 0 <= idx.start < slice_stop_alt(idx)


def format_field_index(idx):
	if isinstance(idx, int):
		return str(idx)

	assert isinstance(idx, slice) and alt_if_none(idx.step, 1) == 1
	return '{:d}-{}'.format(idx.start, alt_if_none(idx.stop, ''))


def str_unescape(s):
	return codecs.escape_decode(s.encode())[0].decode()


def parse_groups(s):
	return tuple(map(parse_field_index, s.split(',')))


def parse_args(args):
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument('groups',
		type=parse_groups,
		help='A list of field indexes or column ranges used to group records '
			'(zero-based, comma-separated).')
	ap.add_argument('aggregations',
		nargs='+', type=Aggregation,
		help='A field index (zero-based) or column range, the name of an '
			'aggregation function ({:s}), and optionally a format string, all '
			'colon-separated.'
			.format(', '.join(Aggregation.aggregator_functions)))
	ap.add_argument('-F', '--input-field-separator', metavar='SEP',
		help='The input field separator string. (default: a series of white-space '
			'characters)')
	ap.add_argument('-O', '--output-field-separator', metavar='SEP',
		type=str_unescape,
		help='The output field separator string. (default: the input field '
			'separator if set and no regular expression, otherwise the tab '
			'character)')
	ap.add_argument('--ifs-regexp', metavar='flags',
		nargs='?', const=0, type=parse_regex_flags,
		help='Interprete the input field separator as a regular expression.')
	ap.add_argument('--skip', metavar='N',
		type=int, default=0,
		help='Skip N lines at the beginning of the input (e. g. header lines).')
	ap.add_argument('-s', '--sorted', action='store_true',
		help='Indicates that the input records are already sorted.')

	return validate_args(ap, ap.parse_args(args))


def validate_args(ap, args):
	# parse/transform and validate field separators
	if args.input_field_separator == '':
		ap.error('The input field separator must not be empty.')
	if args.ifs_regexp is None:
		if args.input_field_separator:
			args.input_field_separator = str_unescape(args.input_field_separator)
	else:
		args.input_field_separator = re.compile(
			args.input_field_separator, args.ifs_regexp)
		if args.input_field_separator.fullmatch(''):
			ap.error('The input field separator must not match the empty string.')
	if args.output_field_separator is None:
		args.output_field_separator = (
			(args.ifs_regexp is None and args.input_field_separator) or '\t')

	# validate groupings and aggregations
	idx_getter = attrgetter('field_index')
	err = tuple(itertools.filterfalse(validate_field_index,
		itertools.chain(args.groups, map(idx_getter, args.aggregations))))
	if err:
		ap.error(
			'Invalid field indices: ' + ', '.join(map(format_field_index, err)))

	if len(args.groups) + len(args.aggregations) > 1:
		err = frozenset(map(type,
			itertools.chain(args.groups, map(idx_getter, args.aggregations))))
		if len(err) > 1:
			ap.error('Cannot mix field indices and column ranges')

	use_column_ranges = isinstance(args.aggregations[0].field_index, slice)
	if use_column_ranges:
		if args.input_field_separator is not None:
			print(ap.prog, 'Warning',
				'The input field separator has no effect in column ranges mode.',
				sep=': ', file=sys.stderr)
		args.input_field_separator = False

	if len(args.groups) > 1:
		# Discard keys and keep only the groups.
		err = map(tuple, map(itemgetter(1),
			itertools.groupby(sorted(args.groups))))
		# Discard single-size groups. Field indices are supposedly equal within
		# their group, so keep only the first.
		err = tuple(group[0] for group in err if len(group) > 1)
		if err:
			ap.error(
				'Duplicate grouping fields: ' + ', '.join(map(format_field_index, err)))

		err = tuple(filter(args.groups.__contains__,
			map(idx_getter, args.aggregations)))
		if err:
			ap.error(
				'Cannot aggregate grouping fields: ' +
					', '.join(map(format_field_index, err)))

	if use_column_ranges and len(args.groups) + len(args.aggregations) > 1:
		err = sorted(
			itertools.chain(args.groups, map(idx_getter, args.aggregations)),
			key=lambda r: (r.start, slice_stop_alt(r)))
		err = [
			' and '.join(map(format_field_index, (a, b)))
			for a, b in zip(err, itertools.islice(err, 1, None))
			if a != b and (a.stop is None or b.start is None or b.start < a.stop)
		]
		if err:
			ap.error('Non-equal overlapping field column ranges: ' + ', '.join(err))

	return args


def main(*args):
	args = parse_args(args or None)

	_print = fpartial(print, sep=args.output_field_separator)
	records = sys.stdin
	if args.skip > 0:
		records = itertools.islice(records, args.skip, None)
	if args.input_field_separator is None:
		records = map(str.split, records)
	else:
		records = map(methodcaller('rstrip', '\n'), records)
		if isinstance(args.input_field_separator, RegexType):
			records = map(args.input_field_separator.split, records)
		elif args.input_field_separator is not False:
			records = map(
				fpartial(str.split, sep=args.input_field_separator), records)
	records = filter(None, records)

	if not args.groups:
		records = ((), records)
	else:
		group_key_func = itemgetter(*args.groups)
		if not args.sorted:
			records = sorted(records, key=group_key_func)
		records = itertools.groupby(records, group_key_func)

	reuse_records = len(args.aggregations) > 1
	if reuse_records:
		reuse_records = map(attrgetter('field_index'), args.aggregations)
		if args.input_field_separator is False:
			reuse_records = map(attrgetter('start', 'stop'), reuse_records)
		reuse_records = len(args.aggregations) != len(frozenset(reuse_records))

	for group_key, group_records in records:
		if reuse_records:
			group_records = tuple(group_records)

		aggregated_fields = (
			format(agg.aggregator(map(itemgetter(agg.field_index), group_records)),
				agg.format)
			for agg in args.aggregations)

		if len(args.groups) == 1:
			_print(group_key, *aggregated_fields)
		else:
			_print(*tuple(itertools.chain(group_key, aggregated_fields)))


if __name__ == '__main__':
	main()
