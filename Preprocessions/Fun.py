#!/usr/bin/python
import pickle

def GetNeighbors(asn):
    if (type(asn) == type('a')):
        name = './MyData/Neighbors/' + asn
    else:
        name = './MyData/Neighbors/%d'%asn
    
    f = open(name, 'r')
    line = f.readline()
    if (line[0]=='#'):
        line = f.readline()

    ProviderList = line.split()  
    line = f.readline()
    CustomerList = line.split()
    line = f.readline()
    PeerList = line.split()

    return [ProviderList, CustomerList, PeerList]



def GetPrefix(asnList):
    # Input a list ASN
    # Out put a list of IP prefixes list, and every list is corresponding to the ASN list
    
    f = open('./MyData/Pickle/ASList.txt', 'r')
    ASList = pickle.load(f)
    f.close()
    f = open('./MyData/Pickle/PrefixList.txt', 'r')
    PrefixList = pickle.load(f)
    f.close()

    # Save results
    asnPrefixList = []    
    for ASN in asnList:
        if (type(ASN) == type('a')):
            ASN_str = ASN
        else:
            ASN_str = str(ASN)
        
        if (not ASN_str in ASList):
            print 'Error, ASN('+ ASN_str +') cannot be found in prefix.txt'
            # Empty list for this situation
            asnPrefixList.append([])
        else:        
            i = ASList.index(ASN_str)
            asnPrefixList.append(PrefixList[i])
              
        #print 'i=%d' %i
        #print 'len(ASList)=%d' %len(ASList)
        #print 'len(PrefixList)=%d' %len(PrefixList)
        #print 'ASN: %d\nIP prefix Number: %d' %(ASN, len(PrefixList[i]))

    print 'returned len of list: %d' %len(asnPrefixList)
    return asnPrefixList    








    
                                            
