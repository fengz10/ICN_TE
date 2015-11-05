import sys
import matplotlib.pyplot as plt
from numpy import arange
from Algo import *

#if len(sys.argv) < 2:
#    exit(1)


revenue0, revenue1, revenue2, revenue3 = main(1)
print 'revenue0=%d'%revenue0
print 'revenue1=%d'%revenue1
print 'revenue2=%d'%revenue2
print 'revenue3=%d'%revenue3

'''
c1 = []
c2 = []
c3 = []

for i in range(10, 19):
    revenue0, revenue1, revenue2, revenue3 = main(i)
    c1.append(revenue0)
    c2.append(revenue2)
    c3.append(revenue3)    
print '-------------------------------------------------'
print '%e'%c1
print '%e'%c2
print '%e'%c3



c1=[374802.62418938376, 189165.53359210223, 537568.7908717527, 322082.2681797495, 182533.7841416931, 71008.02578351473, 337531.88557456183, 107402.82551447971, 540993.5049594609]
c2=[579599.8571280716, 423449.2992367267, 761619.7918176365, 584727.7642735775, 496938.6533120666, 393906.67559847084, 630394.2480057303, 496134.7756647193, 851035.9118496866]
c3=[591249.6893292187, 495283.36942455865, 778123.3887495884, 703568.8571708098, 468523.32777155266, 428212.3304403171, 670140.9179129376, 445933.7995247246, 861373.213060431]


x1 = range(10, 19)
#x1 = arange(0.1, 1.1, 0.1)
plt.plot(x1, c1, 'g^-', label='Non-cooperative')
plt.plot(x1, c2, 'bo-', label='Cooperative')
plt.plot(x1, c3, 'r*-', label='Integrated optimization')
plt.xlabel('Storage size')
plt.ylabel('ISP revenue')
plt.legend(loc='upper left', numpoints = 1)
plt.show()

'''


