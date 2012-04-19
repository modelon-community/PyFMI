
class Analyze(object):

	def __init__(self,logger):
		"""
		Creator, takes a logger from the FMUModel class
		as input
		"""
		
		# Set flags for filtering
		self._timing_signature = 'TIMING'
		self._debug_signature = 'DEBUG'
		
		self.logger = logger
		
		# Partition logger entries by signature
		self._timing = []
		self._debug = []
		
		for entry in self.logger:
			signature = entry[2]
			info = entry[3].split()
			if signature == self._timing_signature:			
				self._timing.append([info[0],info[2],info[5]])
				
			elif signature == self._debug_signature:
				info_dict = dict([(info[1], info[2]),
								  (info[4], info[5]),
								  (info[6], info[7]),
								  (info[8], info[9]),
								  (info[10],info[11])
								 ])
				self._debug.append([info[0],info_dict])
		
	def total_time(self):
		"""
		Prints the total time spent in nl-solver 
		of blocks
		"""
		tt = 0.0
		
		for entry in self._timing:
			tt += float(entry[2])
		
		return tt
		
	def get_debug(self,identifier= None):
		"""
		Get debug info for blocks
		"""
		if identifier == None:
			print "Possible identifiers are:"
			print self._debug[0][1].keys()
		
		else:
			try:
				nb_ids = identifier.__len__()
				ids = identifier
			except AttributeError:
				# Not a list, make it a list
				ids = [identifier]
				nb_ids = 1
			
			res = []
			try:
				for entry in self._debug:
					res_tmp = [entry[0]]
					for i in ids:
						res_tmp.append(entry[1][i])
					res.append(res_tmp)
				
			except KeyError:
				print "Identifier ", identifier, " not present in debug info"
				print "Possible identifiers are:"
				print self._debug[0][1].keys()
		
			return res
