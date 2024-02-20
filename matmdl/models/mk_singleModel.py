"""
Creates input files for multiple vertical elements.
Jan 31, 2020.
"""
import os 


def main():
	num_el = ask_dimensions()
	mk_singleModel(num_el)


def ask_dimensions():
    """
    Get user input for model dimensions.

    Args:

    Returns:
        number_of_elements (int): Length of model in the y (loading) direction

    """
    number_of_elements = int(input('Number of elements: '))
    return number_of_elements


def mk_singleModel(number_of_elements):
	"""
		Make a chain of elements for single crystal modeling.

		Args:
			number_of_elements (int): Length of chain model along loading
				direction, which is y

		Returns:

	"""

	# name = 'Mesh_' + str(number_of_elements) + '_elements.inp'
	name = 'Mesh_Xelement.inp'
	try:
		os.remove(name)
	except:
		pass

	f = open(name, 'w+')

	# write nodes 
	f.write(
	'*Node, nset=All\n'+
		'\t1,\t0.,\t0.,\t0.\n' +
		'\t2,\t1.,\t0.,\t0.\n' +
		'\t3,\t1.,\t1.,\t0.\n' +
		'\t4,\t0.,\t1.,\t0.\n' +
		'\t5,\t0.,\t0.,\t1.\n' +
		'\t6,\t1.,\t0.,\t1.\n' +
		'\t7,\t1.,\t1.,\t1.\n' +
		'\t8,\t0.,\t1.,\t1.\n'
	)
	# n is each additional layer
	for n in range(2, number_of_elements + 1):
		if n == number_of_elements:
			f.write(
				'\t' + str(n*4 + 1) + ',\t1.,\t' + str(n) + '.,\t0.\n' +
				'\t' + str(n*4 + 2) + ',\t0.,\t' + str(n) + '.,\t0.\n' +
				'\t' + str(n*4 + 3) + ',\t1.,\t' + str(n) + '.,\t1.\n' +
				'\t' + str(n*4 + 4) + ',\t0.,\t' + str(n) + '.,\t1.'
			)
		else:
			f.write(
				'\t' + str(n*4 + 1) + ',\t1.,\t' + str(n) + '.,\t0.\n' +
				'\t' + str(n*4 + 2) + ',\t0.,\t' + str(n) + '.,\t0.\n' +
				'\t' + str(n*4 + 3) + ',\t1.,\t' + str(n) + '.,\t1.\n' +
				'\t' + str(n*4 + 4) + ',\t0.,\t' + str(n) + '.,\t1.\n'
			)
	if number_of_elements ==1:
		f.write('**')
	# get correct ordering of nodes
	def order_next(prev_order, element_number):
		max_nodes = 8 + 4 * (element_number - 1)
		next_order = [	prev_order[3], prev_order[2], max_nodes - 3, max_nodes - 2,
						prev_order[7], prev_order[6], max_nodes - 1, max_nodes - 0]
		return next_order

	# order of first element:
	order = [1,2,3,4,5,6,7,8]

	# initiate nodesets with nodes from the bottom of the first element
	nset_left = [1,5]
	nset_right = [2,6]
	nset_front = [5,6]
	nset_back = [1,2]

	# write elements in terms of nodes, in correct order:
	for n in range(1, number_of_elements + 1):
		# write heading for element
		f.write(
		'\n*Element, type=C3D8, elset=All\n' + 
		str(n) + ', '
		)

		# write out node order
		for i in range(0, 7):
			f.write(str(order[i]) + ', ' )
		f.write(str(order[7]))

		# add top nodes to appropriate nodesets
		nset_left.append(order[3])
		nset_left.append(order[7])

		nset_right.append(order[2])
		nset_right.append(order[6])

		nset_front.append(order[6])
		nset_front.append(order[7])

		nset_back.append(order[2])
		nset_back.append(order[3])

		# update node order 
		order = order_next(order, n+1)

	# modification to consider only top and bottom elements 
	# in nset definitions (following 4 lines):
	nset_left = nset_left[:2] + nset_left[-2:]
	nset_right = nset_right[:2] + nset_right[-2:]
	nset_front = nset_front[:2] + nset_front[-2:]
	nset_back = nset_back[:2] + nset_back[-2:]

	## write node sets for faces 
	def write16perLine(items):
		length = len(items)
		for i in range(length):
			if i == length - 1:
				f.write(str(items[i]))
			elif ((i+1)%16 == 0) and (i != 0):
				f.write(str(items[i]) + ',\n')
			else:
				f.write(str(items[i]) + ', ')

	# left
	f.write('\n*Nset, nset=Left\n')
	write16perLine(nset_left)

	# right
	f.write('\n*Nset, nset=Right\n')
	write16perLine(nset_right)

	# front
	f.write('\n*Nset, nset=Front\n')
	write16perLine(nset_front)

	# back
	f.write('\n*Nset, nset=Back\n')
	write16perLine(nset_back)

	# top
	if number_of_elements == 1:
		nset_top = [3, 4, 7, 8]
	else:
		total_nodes = 8 + 4 * (number_of_elements - 1)
		nset_top = [total_nodes, total_nodes -1, total_nodes - 2, total_nodes -3]
	f.write('\n*Nset, nset=Top\n')
	for i in range(4):
		f.write(str(nset_top[i]) + ', ')

	# bottom
	f.write(
		'\n*Nset, nset=Bottom\n' +
		'1, 2, 5, 6'
	)

	## write end matter 
	f.write(
		'\n**\n' + 
		'*Parameter\n' +
		'x=1.\n' + 
		'y=' + str(number_of_elements) + '.\n' +
		'z=1.\n' + 
		'x_Half=x/2\n' +
		'y_Half=y/2\n' +
		'z_Half=z/2\n' +
		'*Node\n' +
		'20000000,\t<x_Half>,\t<y_Half>,\t0.\n' +
		'20000001,\t<x_Half>,\t<y_Half>,\t<z_Half>\n' +
		'20000002,\t<x_Half>,\t<y>,\t\t0.\n' +
		'20000003,\t<x>,\t\t<y_Half>,\t0.\n' +
		'20000004,\t0.,\t\t<y_Half>,\t0.\n' +
		'20000005,\t<x_Half>,\t0.,\t\t0.\n' +
		'*Nset, nset=RP-Back\n' +
		'20000000\n' +
		'*Nset, nset=RP-Front\n' +
		'20000001\n' +
		'*Nset, nset=RP-Top\n' +
		'20000002\n' +
		'*Nset, nset=RP-Right\n' +
		'20000003\n' +
		'*Nset, nset=RP-Left\n' +
		'20000004\n' +
		'*Nset, nset=RP-Bottom\n' +
		'20000005\n'
	)
	f.close()

if __name__ == '__main__':
    main()