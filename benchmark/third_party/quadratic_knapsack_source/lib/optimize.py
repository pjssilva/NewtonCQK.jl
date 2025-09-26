'''
Generate an optimized version of the continuous knapsack solvers.

There are three possible optimizations:

--set_d=value: sets fixed value for the Hessian diagonal (that defines the
               norm), avoiding unecessary fetches and even operations if the
               compiler can optimize it out.

--set_b=value: sets fixed value for the normal of the hyperplane, avoiding
               unecessary fetches and even operations if the compiler can
               optimize it out.

--no_compression: Turn off compression in Newton method.

Copyright: Paulo J. S. Silva <pjssilva@gmail.com> 2013.
'''

from optparse import OptionParser
import re

files = ['cont_quad_knapsack.h', 'cont_quad_knapsack.c',
         'third_party_methods.h', 'third_party_methods.c']


def get_options():
    parser = OptionParser()
    parser.add_option('--set_d', type='string', dest='d',
                      help='set fixed value for Hessian diagonal')
    parser.add_option('--set_b', type='string', dest='b',
                      help='set fixed value for the normal of the hyperplane')
    parser.add_option('--no_compression', action='store_false', default='True',
                      dest='compression',
                      help='turn off compression (variable fixing) in Newton')
    return parser.parse_args()


def set_d(input, value):
    '''
    Replace ->d[.*?] to value.
    '''
    d_re = re.compile('p->d\[.*?\]')
    return d_re.sub(value, input)


def set_b(input, value):
    '''
    Replace ->b[.*?] to value.
    '''
    b_re = re.compile('p->b\[.*?\]')
    return b_re.sub(value, input)


def cut_off_slopes(input, d_value, b_value):
    '''
    Cut off the slopes vector from the code, it is constant equal 1.
    '''
    # Get rid of lines that define the values of the slopes vector. It
    # is always constan equal 1.0.
    slopes_re_left_hand = re.compile('slopes\[.*?\]\s*?=')
    res = ''
    input = input.split('\n')
    for line in input:
        if slopes_re_left_hand.search(line):
            print 'Deleting line', line
            continue
        res += line + '\n'

    # Get rid of the access to retrieve the value of the slopes
    # vector, it is always 1.
    slopes_re = re.compile('slopes\[.*?\]')
    value = '%.16e' % (float(b_value)**2/float(d_value))
    return slopes_re.sub(value, res)


def turn_off_compression(input):
    '''Delete call to compression in Newton's method and all appearances of the
    indirection vector ind.
    '''
    ind_re = re.compile('ind\[.*?\]')
    fix_call_re = re.compile('\s*newton_fix\(')
    res = ''
    input = input.split('\n')
    for line in input:
        if fix_call_re.match(line) or ind_re.search(line):
            print 'Deleting line', line
            continue
        res += line + '\n'

    # Fix the indeixing to use the direct index.
    return res.replace('[ii]', '[i]')


def optimize_file(file_name, options):
    '''Read a source file and apply required optimizations.
    '''
    # Read source
    content = file(file_name).read()

    # Optimize source
    if options.d:
        content = set_d(content, options.d)
    if options.b:
        content = set_b(content, options.b)
    if options.d and options.d:
        content = cut_off_slopes(content, options.d, options.b)
    if not options.compression and file_name.startswith('cont_quad_knapsack'):
        content = turn_off_compression(content)

    # Save optimized source
    file('opt_' + file_name, 'w').write(content)


def main():
    options, args = get_options()
    for file_name in files:
        optimize_file(file_name, options)

if __name__ == '__main__':
    main()
