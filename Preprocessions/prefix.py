#!/usr/bin/python
import string
import pickle

# Calculate the IP prefixes list for every AS, and output it to files
ASList = []
# ASListCare = [1239, 3356, 7018, 8002, 25973, 14744, 5400, 3209, 7713]
# Create a IP prefixes list for every AS
PrefixList = []

#for ASN in ASList:
#    PrefixList.append([])

f = open('./OriginData/routeviews.pfx2as')
n = 0
while 1:
    line = f.readline()
    if not line:
        break
    
    #if (line[0] == '#'):
    #    continue
    # The format of every line is: IP prefix, prefix length, and owner AS number (splitted by comma, if more than one)
    line = line.split()
    #print line[2]
    #for i in range(len(ASList)):
        #pass
        #print ASN
        #print 'i = ' + str(i)
        
    line2_ASList = line[2].split(',')
        #if (len(line2_ASList) > 1):            
        #    print len(line2_ASList)
    # If the AS number is already in the list, add the prefix to the prefixes list
    # Else, add the AS number to the AS list, and the prefix to the corresponding prefix list
    for ASN in line2_ASList:
        if ASN in ASList:
            PrefixList[ASList.index(ASN)].append(line[0] +'/'+ line[1])
        else:
            ASList.append(ASN)
            PrefixList.append([])
            PrefixList[len(ASList)-1].append(line[0] +'/'+ line[1])
            
    #print line[0]
    #print line[1]
    #print line[2]

    #n += 1
    #if (n == 20):
    #   break
f.close()
f_pickle = open('./MyData/Pickle/ASList.txt', 'w')
pickle.dump(ASList, f_pickle)
f_pickle.close()

f_pickle = open('./MyData/Pickle/PrefixList.txt', 'w')
pickle.dump(PrefixList, f_pickle)
f_pickle.close()

print '-----------------------------'
print 'AS number:'
print len(ASList)


print '------------------------------'
#for ASN in ASListCare:
#    i = ASList.index(str(ASN))
#    print 'IP prefix number of ' + str(ASN) +':'
#    print len(PrefixList[i])
#    print '------------------------------'
#    fo = open('./as2prefix/'+str(ASN), 'w')
#    for prefix in PrefixList[i]:
#        fo.write(prefix[0])
#        fo.write('\t')
#        fo.write(prefix[1])
#        fo.write('\n')
#    fo.close()
             
    
