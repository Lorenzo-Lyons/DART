#!/usr/bin/env python3


class Readline:
	def __init__(self, s):
		self.buf = bytearray()
		self.s = s
		
	def readline(self):
		wait = self.s.in_waiting
		data = self.s.read(wait)
		r = self.buf.extend(data)
		i = self.buf.find(b"\n")
		if i > 0:
			data_split = self.buf.split(b"\n")
			r = data_split[-2]
			self.buf = data_split[-1]
			return r.decode('utf8')
		else:
			return None

