#!/usr/bin/python3 -O
"""Group and aggregate record fields"""

import sys
import math
import array
import codecs
import argparse
import itertools
from operator import itemgetter, attrgetter, methodcaller
from functools import reduce, partial as fpartial


try:
	import regex as re
except ImportError:
	import re

try:
	RegexType = re._pattern_type
except AttributeError:
	RegexType = type(re.compile(''))
else:
	assert isinstance(re.compile(''), RegexType)


def pairs(iterable, count=2):
	return zip(*map(itertools.islice,
		itertools.tee(iterable, count), itertools.count(), itertools.repeat(None)))


class sized_map(collections.abc.Iterator):

	__slots__ = ('__next__', '__len__')


	def __init__(self, func, *iterables, size=None):
		self.__next__ = map(func, *iterables).__next__

		if size is None:
			try:
				size = min(map(len, iterables))
			except TypeError:
				pass
		elif not isinstance(size, int):
			raise TypeError("'size' must be an int or None")
		elif size < 0:
			raise ValueError("'size' must not be negative")
		self.__len__ = None if size is None else lambda: size



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


def iavg(iterable, default=float('nan')):
	iterable = sized_map(int, iterable)
	if iterable.__len__:
		size = len(iterable)
		total = sum(iterable)
	else:
		total, size = reduce(
			lambda acc, x: (acc[0] + x, acc[1] + 1), iterable, (0, 0))
	return total / size if size else default


def favg(iterable, default=float('nan')):
	iterable = sized_map(float, iterable)
	if not iterable.__len__:
		iterable = array.array('d', iterable)
	size = len(iterable)
	return math.fsum(iterable) / size if size else default


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
			for v in zip(slots[2:], args[2:]):
				setattr(self, *v)

		else:
			if l_args < l_mandatory:
				raise TypeError(
					'Too few arguments: {:d} < {:d}'.format(l_args, l_mandatory))
			if l_args > l_slots:
				raise TypeError(
					'Too many arguments: {:d} > {:d}'.format(l_args, l_slots))
			for v in zip(self.__slots__, args):
				setattr(self, *v)

		for k in self.__slots__[l_args:]:
			setattr(self, k, defaults[k])


	def _init_kw_args(self, kwargs):
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


def slice_stop_alt(r, alt=sys.maxsize):
	return alt_if_none(r.stop, alt)


def parse_regex_flags(s):
	flags = 0
	try:
		for c in s:
			flags |= getattr(re, c.upper())
	except (AttributeError, TypeError):
		raise ValueError('Invalid regular expression flag: ' + repr(c))
	return flags


def assert_slice(idx):
	return isinstance(idx, slice) and alt_if_none(idx.step, 1) == 1


def validate_field_index(idx):
	if isinstance(idx, int):
		return idx > 0

	assert assert_slice(idx)
	return 0 <= idx.start < slice_stop_alt(idx)


def format_field_index(idx):
	if isinstance(idx, int):
		return str(idx)

	assert assert_slice(idx)
	return '{:d}-{}'.format(idx.start, slice_stop_alt(idx, ''))


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
		help='Interprete the input field separator as a regular expression. '
			'The optional argument value may contain the following flags: {}.'
			.format(''.join(filter(lambda n: len(n) == 1, re.__all__)).lower()))
	ap.add_argument('-s', '--skip', metavar='N',
		type=int, default=0,
		help='Skip N lines at the beginning of the input (e. g. header lines).')
	ap.add_argument('-S', '--no-sort', dest='sorted',
		action='store_false', default=True,
		help='Assume that the input records are already sorted.')

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
		err = tuple(itertools.chain.from_iterable(
			itertools.islice(g, 1, 2)
			for k, g in itertools.groupby(sorted(args.groups))))
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
		err = [
			' and '.join(map(format_field_index, (a, b)))
			for a, b in pairs(sorted(
				itertools.chain(args.groups, map(idx_getter, args.aggregations)),
				key=lambda r: (r.start, slice_stop_alt(r))))
			if a != b and (a.stop is None or b.start < a.stop)
		]
		if err:
			print(ap.prog, 'Warning', 'Non-equal overlapping field column ranges',
				', '.join(err), sep=': ', file=sys.stderr)

	return args


def process_records(records, opts):
	_print = fpartial(print, sep=opts.output_field_separator)

	reuse_records = len(opts.aggregations) > 1
	if reuse_records:
		reuse_records = map(attrgetter('field_index'), opts.aggregations)
		if opts.input_field_separator is False:
			reuse_records = map(attrgetter('start', 'stop'), reuse_records)
		reuse_records = len(opts.aggregations) != len(frozenset(reuse_records))

	for group_key, group_records in records:
		if reuse_records:
			group_records = tuple(group_records)

		aggregated_fields = (
			format(agg.aggregator(map(itemgetter(agg.field_index), group_records)),
				agg.format)
			for agg in opts.aggregations)

		if len(opts.groups) == 1:
			_print(group_key, *aggregated_fields)
		else:
			_print(*tuple(itertools.chain(group_key, aggregated_fields)))


def main(*args):
	opts = parse_args(args or None)

	records = sys.stdin
	if opts.skip > 0:
		records = itertools.islice(records, opts.skip, None)
	if opts.input_field_separator is None:
		records = map(str.split, records)
	else:
		records = map(methodcaller('rstrip', '\n'), records)
		if isinstance(opts.input_field_separator, RegexType):
			records = map(opts.input_field_separator.split, records)
		elif opts.input_field_separator is not False:
			records = map(
				fpartial(str.split, sep=opts.input_field_separator), records)
	records = filter(None, records)

	if not opts.groups:
		records = ((), records)
	else:
		group_key_func = itemgetter(*opts.groups)
		if not opts.sorted:
			records = sorted(records, key=group_key_func)
		records = itertools.groupby(records, group_key_func)

	return process_records(records, opts)


if __name__ == '__main__':
	main()
