from Fun import GetPrefix
from Fun import GetNeighbors
import pickle


# Generate pikle data for customer and peer ASes of the ASes we care for
ASListCare = [1239, 3356, 7018, 8002, 25973, 14744, 5400, 3209, 7713]


# NeighborsListCare records ASN of its self, providers, customers and peers

NeighborsListCare = []
# Record the number of proviers, customers, and peers, in accordance with ASListCare.
ProviderNumberCare = []
CustomerNumberCare = []
PeerNumberCare = []
for ASN in ASListCare:
    [ProviderList, CustomerList, PeerList] = GetNeighbors(ASN)
    
    ProviderNumberCare.append(len(ProviderList))
    CustomerNumberCare.append(len(CustomerList))
    PeerNumberCare.append(len(PeerList))

    NeighborsListCare.append(ASN) 
    NeighborsListCare.extend(ProviderList)
    NeighborsListCare.extend(CustomerList)
    NeighborsListCare.extend(PeerList)
    
    
# For efficiency, recall GetPrefix here and pickle the results.
AllPrefixCare = GetPrefix(NeighborsListCare)

# Save the results of IP prefixes to pickle to have better performance
# Save according to the original AS number
  

# Generating differnt IP prefixes lists
pos = 0
for i in range(len(ASListCare)):    
    # Read the AS number of it self, only the first value
    PrefixListSelf = AllPrefixCare[pos: pos + 1]
    pos += 1
    # Read providers
    PrefixListProvider = AllPrefixCare[pos: pos + ProviderNumberCare[i]]
    pos += ProviderNumberCare[i]
    # Read customers
    PrefixListCustomer = AllPrefixCare[pos: pos + CustomerNumberCare[i]]
    pos += CustomerNumberCare[i]
    # Read peers
    PrefixListPeer = AllPrefixCare[pos: pos + PeerNumberCare[i]]
    pos += PeerNumberCare[i]
    
    f = open('./MyData/Pickle/as2prefix/NeighborPrefixesAS%d.txt'%ASListCare[i], 'w')
    pickle.dump(PrefixListSelf, f)
    pickle.dump(PrefixListProvider, f)
    pickle.dump(PrefixListCustomer, f)
    pickle.dump(PrefixListPeer, f)
    f.close()

