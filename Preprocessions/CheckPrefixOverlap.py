from Fun import GetPrefix
from Fun import GetNeighbors
import ipaddr
import pickle
import random

ASListCare = [1239, 3356, 7018, 8002, 25973, 14744, 5400, 3209, 7713]

ASN = 14744
[ProviderList, CustomerList, PeerList] = GetNeighbors(ASN)
print ProviderList
print CustomerList
print PeerList

####################Function of judging overlapping###########################
# return 1 for having overlapping
#        0 for not
def IsOverlapPrefixList(ListA, ListB):
    for prefix1 in ListA:
        for prefix2 in ListB:
            if prefix1.overlaps(prefix2):
                return True
    return False

##########Map one prefix to a content ID, the ID value starts from 0###########
# Return ID, -1 if not found
def GetContentID(ClusterList, prefix):
    for i in range(len(ClusterList)):
        if (prefix in ClusterList[i]):
            return i
    print 'Cannot find prefix %s'%prefix
    return -1  
##########################################################################


# Read the IP prefixes from file
f = open('./MyData/Pickle/as2prefix/NeighborPrefixesAS%d.txt'%ASN, 'r')
asnPrefixSelf = pickle.load(f)
asnPrefixProvider = pickle.load(f)
asnPrefixCustomer = pickle.load(f)
asnPrefixPeer = pickle.load(f)
f.close()

# asnPrefixCustomer is the IP prefixes list of customers, the length of the list equals to the number of customers
# asnPrefixPeer
# Change the elements of asnPrefixCustomer and asnPrefixPeer into the format of ipaddr.IPNetwork, for checking longest prefix match

CustomerNum = len(asnPrefixCustomer)
PeerNum = len(asnPrefixPeer)
print 'CustomerNum = %d'%CustomerNum
print 'PeerNum = %d'%PeerNum

for i in range(CustomerNum):
    asnPrefixCustomer[i] = map(ipaddr.IPNetwork, asnPrefixCustomer[i])

for i in range(PeerNum):
    asnPrefixPeer[i] = map(ipaddr.IPNetwork, asnPrefixPeer[i])

# Record IP prefixes of all the customers and peers
allPrefix = []
for asnP in asnPrefixCustomer:
    allPrefix.extend(asnP)

for asnP in asnPrefixPeer:
    allPrefix.extend(asnP)

print 'The number of all the IP Prefix(Customer and Peer) = %d'%len(allPrefix)
if (len(allPrefix) != len(list(set(allPrefix)))):
    print 'The same IP prefix appeared, duplicated prefixes have been deleted'
    allPrefix = list(set(allPrefix))
    print 'len after set = %d'%len(list(set(allPrefix)))

#allPrefix = list(set(allPrefix))
print 'len after set = %d'%len(allPrefix)

# Find out the longest prefix matched IP prefixes
# Treat the longest matched prefixes cluster as a type of content

duplicatePrefix = []

prefixCluster = []
for prefix in allPrefix:
    prefixCluster.append([prefix])

#while (len(prefixCluster) > 50):
#    prefixCluster.pop()

while 1:
    lenAll = len(prefixCluster)
    if (lenAll < 2):
        break
    # Compare any two lists in prefixCluster, combine them if possible
    try:
        for i in range(lenAll-1):
            for j in range(i+1, lenAll):
                if IsOverlapPrefixList(prefixCluster[i], prefixCluster[j]):
                    # If two lists are overlapped, combined them
                    prefixCluster[i].extend(prefixCluster[j])
                    del prefixCluster[j]
                    raise Exception("break")
    except:
        continue
    
    break

print 'No duplicate list number: %d'%len(prefixCluster)
print 'It represents the number of contents is %d'%len(prefixCluster)
# prefixCluster records the prefixes list which has no overlapping, and we trea it as content list
# The length of a list indicates that the content is owned by customers and peers

# Shuffle the contents order

random.shuffle(prefixCluster)

# Construct the content ID, and treat IP prefixes as content IDs
# Each element in CustomerContentList indicates the content IPs of a customer AS
# The same priciple for Peers
CustomerContentList = []
PeerContentList = []

for pCustomer in asnPrefixCustomer:
    ContentCustomer = []
    for p in pCustomer:
        ContentCustomer.append(GetContentID(prefixCluster, p))
    CustomerContentList.append(ContentCustomer)

for pPeer in asnPrefixPeer:
    ContentPeer = []
    for p in pPeer:
        ContentPeer.append(GetContentID(prefixCluster, p))
    PeerContentList.append(ContentPeer)

   
print 'ASN(%d) has %d Customers'%(ASN, len(CustomerContentList))
print 'The list of content of Customers:'
print CustomerContentList

print '-------------------------------------------------------------'
print 'ASN(%d) has %d Peers'%(ASN, len(PeerContentList))
print 'The list of content of Peers:'
print PeerContentList

# Eleminate the same content ID by set structure
CustomerContentList = map(set, CustomerContentList)
#CustomerContentList = map(list, CustomerContentList)
PeerContentList = map(set, PeerContentList)

f = open('./MyData/Pickle/Content/ContentID_%d.txt'%ASN, 'w')
pickle.dump(CustomerContentList, f)
pickle.dump(PeerContentList, f)
f.close()


AllContent = set([])
for c in CustomerContentList:
    AllContent |= c
for c in PeerContentList:
    AllContent |= c

print 'The number of all the content is: %d'%len(AllContent)
    





        
    
    
    


