#!/usr/bin/python
# change the '_' char in pfx2as file into ',' char to ease the call of split function

f = open('routeviews-rv2-20140401-1400.pfx2as')
fo = open('routeviews22.pfx2as', 'w')
while 1:
    c = f.read(1)
    if not c:
        break

    if c == '_':
        fo.write(',')
    else:
        fo.write(c)

f.close()
fo.close()
