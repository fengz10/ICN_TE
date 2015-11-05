#!/usr/bin/python
import string
# Calculate the number of neighbors, and the neighbors list for each AS
ASList = [1239, 3356, 7018, 8002, 25973, 14744, 5400, 3209, 7713,
          1273, 1299, 3356, 3549, 8928, 20562, 21385,
          1273, 2914, 3356, 3549, 3561,
          174, 1299, 2914, 3257, 3356, 6453, 6762, 7473,
          174, 3257,
          174, 2914, 3257, 3549, 6461, 7018,
          3356, 3549, 4436]

print 'len(ASList)=%d'%len(ASList)
print 'After set:'
ASList = list(set(ASList))
print 'len(ASList)=%d'%len(ASList)

#[5580, 5400, 48961, 251, 12390, 1273, 3209, 5089, 3491,
#          35017, 1257, 4323, 38930, 14744, 8928, 25973, 7922,
#          33724, 2711, 8002, 22652, 7713]
# ASResult is the results, and each line for an AS
#[ASN, provider, providerList, customer, customerList, peer, peerList]
ASResult = []

# Note that cannot use equal to connect three list variables in a statement, since it will only produce one list
peerList = []
providerList = []
customerList = []
peer = provider = customer = 0


Datalist = []
# Read the original relationship file
f = open('./OriginData/20131101.as-rel.txt')
while 1:
    line = f.readline()
    if not line:
        break
    if (line[0] == '#'):
        continue

    line = line.split('|')
    Datalist.append([string.atoi(line[0]), string.atoi(line[1]),
                      string.atoi(line[2])])
f.close()

# for each item in ASList, compute its AS relationship

for ASN in ASList:
    for ASData in Datalist:
        if ((ASN == ASData[0]) and (ASData[2] == -1)):
            customer += 1
            customerList.append(ASData[1])
        elif ((ASN == ASData[0]) and (ASData[2] == 0)):
            peer += 1
            peerList.append(ASData[1])
        elif ((ASN == ASData[1]) and (ASData[2] == -1)):
            provider += 1
            providerList.append(ASData[0])
        elif ((ASN == ASData[1]) and (ASData[2] == 0)):
            peer += 1
            peerList.append(ASData[0])
        else:
            pass

    ASResult.append([ASN, provider, providerList, customer, customerList,
                   peer, peerList])
    
    peer = provider = customer = 0
    peerList = []
    providerList = []
    customerList = []
 
for eachAS in ASResult:
    print 'AS number: ' + str(eachAS[0])
    print 'provider number: ' + str(eachAS[1])
    print 'provider AS: '
    #print eachAS[2]
    print 'customer number: ' + str(eachAS[3])
    print 'customer AS: '
    #print eachAS[4]
    print 'peer number: ' + str(eachAS[5])
    print 'peer AS: '
    #print eachAS[6]  
    print '-----------------------------------'

    # Record all the providers and customers, and use the AS number as the file name
    f = open('./MyData/Neighbors/' + str(eachAS[0]), 'w')
    f.write('#The following three lines enumerate the list of providers, ' +
              'customers, and peers\n')
    # This line is provider list, splitted by a blank space
    for eachProvider in eachAS[2]:
        f.write(str(eachProvider))
        f.write(' ')        
    f.write('\n')
    # This line is for customers
    for eachCustomer in eachAS[4]:
        f.write(str(eachCustomer))
        f.write(' ')
    f.write('\n')
    # This line is for peers
    for eachPeer in eachAS[6]:
        f.write(str(eachPeer))
        f.write(' ')
    f.write('\n')

    f.close()

ASN_list = []
for ASData in Datalist:
# Record all the AS number
    ASN_list.append(ASData[0])
    ASN_list.append(ASData[1])
    
print 'Before eliminating the same elements：'
print len(ASN_list)
print 'After eliminating the same elements：'
print len(list(set(ASN_list)))


