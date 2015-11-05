import numpy as np
from numpy import array
import math
import copy
import random
from Fun import *
import pickle
import random
from numpy import arange
import matplotlib.pyplot as plt
from pulp import *
import time
import bisect
import math

# Replicate locally is regarded as a neighbor AS and its price is epsilon
epsilon = -1e-05 

############################Zipf Generater####################################
class ZipfGenerator:
    def __init__(self, n, alpha): 
        # Calculate Zeta values from 1 to n: 
        tmp = [1. / (math.pow(float(i), alpha)) for i in range(1, n+1)] 
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0]) 

        # Store the translation map: 
        self.distMap = [x / zeta[-1] for x in zeta] 

    def next(self): 
        # Take a uniform 0-1 pseudo-random value: 
        u = random.random()  

        # Translate the Zipf variable: 
        return bisect.bisect(self.distMap, u) - 1


######################Dynamic programming for knapsack problem############################
def knapsack01_dp(items, limit):
    table = [[0 for w in range(limit + 1)] for j in xrange(len(items) + 1)]
    for j in xrange(1, len(items) + 1):
        item, wt, val = items[j-1]
        for w in xrange(1, limit + 1):
            if wt > w:
                table[j][w] = table[j-1][w]
            else:
                table[j][w] = max(table[j-1][w], table[j-1][w-wt] + val)

    result = []
    w = limit
    for j in range(len(items), 0, -1):
        was_added = table[j][w] != table[j-1][w]

        if was_added:
            item, wt, val = items[j-1]
            result.append(items[j-1])
            w -= wt

    return result

#######################Dynamic programming for knapsack problem########################
def knapsack_dp(items, limit):
    table = np.zeros((len(items) + 1, limit + 1), dtype=np.int)
    #table = [[0 for w in range(limit + 1)] for j in xrange(len(items) + 1)]
    #table = array(table)   
 
    for j in range(1, len(items) + 1):
        item, wt, val = items[j-1]
        for w in range(1, limit + 1):
            if wt > w:
                table[j][w] = table[j-1][w]
            else:
                table[j][w] = max(table[j-1][w],
                                  table[j-1][w-wt] + val) 
    result = []
    w = limit
    for j in range(len(items), 0, -1):
        was_added = table[j][w] != table[j-1][w]
 
        if was_added:
            item, wt, val = items[j-1]
            result.append(items[j-1])
            w -= wt
 
    return result
###########################################################################
def knapsack_FPTAS(items, limit, delta=0.01):
    items = array(items)
    P = items[:,2].max()
    K = delta * P/len(items)
    
    for j in range(1, len(items) + 1):
        item, wt, val = items[j-1]
        items[j-1][2] = int(math.floor(val/K))

    return knapsack_dp(items, limit)
############################################################################

##############################Algorithm 1###############################
# Output the profit and bandwidth usage
def algo1(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size):
    # Since we change some variables in the function, such as Storage, BW，and T_avail.
    # It needs to deep copy it to make sure the call of the function will not change the value of input variables
    # Even tuple type can be changed, if it has a list element.

    BW = list(BW)
    T_avail = list(copy.deepcopy(T_avail))
    global epsilon

    # t[N] records the TE strategy, -1 for local replcate, -2 for default, and other value indicates its neighbor AS ID
    t = [-2] * N
    # x records content replicated locally
    x = set([])

    # Cally the dynamic programming to compute local replicate strategy
##    items = []
##    for n in range(N):
##        items.append((n, int(Size[n]), Size[n] * RequestNumOfID[n]))
##    StoredItems = knapsack01_dp(items, Storage)
##    # Content replicated are all in StoredItems
##    for item in StoredItems:
##        x.add(item[0])
     

    # Sort the data from large to small, and record the subscript by k    
    k = (-np.array(RequestNumOfID)).argsort()
    # Replicate greedily
    for i in k:
        if Storage - Size[i] >= 0:
            x.add(i)
            Storage -= Size[i]         
    # Do not consider local replication    
    # Sort price from large to small
    k = (-np.array(P)).argsort()
    assert len(k) == M

    for j in k: 
        # Reachable neighbors list
        ContentAvailList = list(T_avail[j])
        if len(ContentAvailList) == 0:
            continue        
        ReqSize = []
        for id in ContentAvailList:
            ReqSize.append(RequestNumOfID[id] * Size[id])
                
        # Sort the bandwidth consumption of different contents from large to small     
        kContentAvail = (-np.array(ReqSize)).argsort()                 
        # The bottleneck of other neighbors is in bandwidth
        for i in kContentAvail:    
            if ReqSize[i] > 0 and BW[j] - ReqSize[i] >= 0:                
                t[ContentAvailList[i]] = j
                BW[j] -= ReqSize[i]
                # Note, delete the content from other neighbors, otherwise it produces a bug
                for s in T_avail:
                    s.discard(ContentAvailList[i])

    # The content discovery in this algorithm is by chance, neither telling TE module definitely, nor without TE (bad performance)
    for i in x:
        if t[i] >= 0 and P[t[i]] < 0:
            # Save the bandwidth
            BW[t[i]] += RequestNumOfID[i] * Size[i]
            t[i] = -1            
        elif t[i] == -2 and RequestNumOfID[i] > 0:
            t[i] == -1
        else:
            pass                

    # Calculate profit
    profit = 0
    for i in range(N):
        if t[i] >= 0:
            profit += RequestNumOfID[i]*Size[i]*P[t[i]]
            
    # Calculate bandwidth usage
    # averageUtiBW = 1 - sum(BW)/(M*4e4)
    [ProviderList, CustomerList, PeerList] = GetNeighbors(ASN)
    # Calculate the bandwidth usage of providers
    # If it is a Tier-1 AS, calculate the bandwidth usage of its peers
    ProviderNum = len(ProviderList)
    PeerNum = len(PeerList)
    
    if ProviderNum > 0:        
        averageUtiBW_P = 1-sum(BW[:ProviderNum])/(ProviderNum*4e4)
    elif PeerNum > 0:
        averageUtiBW_P = 1-sum(BW[PeerNum:])/(PeerNum*4e4)
    else:
        print 'No Providers and Peers'       
    
            
    # Statistics of the contents which cannot be served since the full of providers' bandwidth
    NotContentList = []
    for i in range(N):
        if RequestNumOfID[i] > 0:
#            print 'Requesting content %d for %d times.'%(i, RequestNumOfID[i])
#            print 'Content %d can be get from neighbor %d'%(i, t[i])            
#            print 'Price of neighbor %d is %d'%(t[i], P[t[i]])
            if t[i] == -2:
                NotContentList.append(i)

    if len(NotContentList) > 0:
        print 'Number of Contents not found in Algo1 is: %d'%len(NotContentList)
        print 'Remain BW of the first 10 providers of Algo1 are:'
        print 'Not found content ID:'
        if len(NotContentList) > 10:
            print NotContentList[:10]
        else:
            print NotContentList
        
        print BW[:10]
           
    return profit, averageUtiBW_P


##############################Algorithm 2################################
# Output the profit and bandwidth usage
def algo2(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size):
    # deep copy the input varibles which will be changed in the funciton
    BW = list(BW)
    T_avail = list(copy.deepcopy(T_avail))

    # t[N] records the TE strategy, -1 for local replcate, -2 for default, and other value indicates its neighbor AS ID
    t = [-2] * N  
    # x records content replicated locally
    x = set([])
    # Sort price from large to small
    k = (-np.array(P)).argsort()
    assert len(k) == M

    for j in k:
        # Content list which can be reached from neighbors
        ContentAvailList = list(T_avail[j])
        if len(ContentAvailList) == 0:
            continue        
        ReqSize = []
        for id in ContentAvailList:
            ReqSize.append(RequestNumOfID[id] * Size[id])
                
        # Sort the bandwidth consumption of different contents from large to small    
        kContentAvail = (-np.array(ReqSize)).argsort() 
        # The bottleneck of other neighbors is in bandwidth
        for i in kContentAvail:
            if ReqSize[i] > 0 and BW[j] - ReqSize[i] >= 0:                
                t[ContentAvailList[i]] = j
                BW[j] -= ReqSize[i]
                # Note, delete the content from other neighbors, otherwise it produces a bug
                for s in T_avail:
                    s.discard(ContentAvailList[i])

    # Replicate contents according to their prices
    # Some contents cannot be obtained, since its requests are huge and any neighbor can not satisfy the bandwidth requirement. So replicate them locally in priority
    BigValue = 1e6   # Treat the contents cannot be obtainted by TE as highest price

    # Calculate prices of different contents
    profitOfID = []
    for i in range(N):
        if t[i] >= 0:
            profitOfID.append(Size[i] * RequestNumOfID[i] * P[t[i]])
        elif t[i] == -2 and RequestNumOfID[i] > 0:            
            profitOfID.append(-Size[i] * RequestNumOfID[i] * BigValue)
        else:
            profitOfID.append(0)
    # Sort the profit from small to large, and the small ones may be minus
    k = np.array(profitOfID).argsort()
    # Greedily store the minus ones
    for i in k:
        if Storage - Size[i] >= 0 and profitOfID[i] < 0:
            x.add(i)
            Storage -= Size[i]
            # Save the bandwidth, so add it accordingly
            if t[i] >= 0:
                BW[t[i]] += RequestNumOfID[i] * Size[i]
            t[i] = -1
            

    # Calculate profit
    profit = 0
    for i in range(N):
        if t[i] >= 0:
            profit += RequestNumOfID[i]*Size[i]*P[t[i]]

    # Calculate bandwidth usage
    #averageUtiBW = 1 - sum(BW)/(M*4e4)
    [ProviderList, CustomerList, PeerList] = GetNeighbors(ASN)
    # Calculate bandwidth usage of providers. For Tier-1, calculate its peers.
    ProviderNum = len(ProviderList)
    PeerNum = len(PeerList)
    
    if ProviderNum > 0:        
        averageUtiBW_P = 1-sum(BW[:ProviderNum])/(ProviderNum*4e4)
    elif PeerNum > 0:
        averageUtiBW_P = 1-sum(BW[PeerNum:])/(PeerNum*4e4)
    else:
        print 'No Providers and Peers'

    # Record the list of contents which cannot be obtained from providers due to the limit of their bandwidth
    NotContentList = []
    for i in range(N):
        if RequestNumOfID[i] > 0:
#            print 'Requesting content %d for %d times.'%(i, RequestNumOfID[i])
#            print 'Content %d can be get from neighbor %d'%(i, t[i])            
#            print 'Price of neighbor %d is %d'%(t[i], P[t[i]])
            if t[i] == -2:
                NotContentList.append(i)

    if len(NotContentList) > 0:
        print 'Number of Contents not found in Algo2 is: %d'%len(NotContentList)
        print 'Remain BW of the 10 providers of Algo2 are:'
        print BW[:10]
            
    return profit, averageUtiBW_P

##############################Algorithm 3################################
# Output the profit and bandwidth usage
def algo3(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size):
    # deep copy the input varibles which will be changed in the funciton   
    BW = list(BW)
    T_avail = list(copy.deepcopy(T_avail))
    global epsilon

    # t[N] and x are TE strategy and replicate strategy respectively
    t = [-2] * N
    x = set([])

    # Treat local replication as another AS, i.e. AS[M+1], and its price is a minus number nearly zero (higher priority than peers)
    P = list(P)
    P.append(epsilon)
    T_avail.append(set(range(N)))

    # Sort the price from large to small
    k = (-np.array(P)).argsort()
    assert len(k) == M+1
    for j in k:
        # Contents list which can be reached from neighbors
        ContentAvailList = list(T_avail[j])
        if len(ContentAvailList) == 0:
            continue        
        ReqSize = []
        for id in ContentAvailList:
            ReqSize.append(RequestNumOfID[id] * Size[id])
            
        # Sort the bandwidth consumption of contents from large to small      
        kContentAvail = (-np.array(ReqSize)).argsort()
        if j == M:
            # Call the dynamic programming routine to decide which contents to replicate
            # The contents needs to replicate are in kContentAvail
            items = []
            for n in range(len(ContentAvailList)):
                # If call the dynamic programming, the value of Size needs to be an integer
                items.append((ContentAvailList[n], Size[ContentAvailList[n]], 
                              ReqSize[n]))
            # Note, compare performance with gurobi, use FPTAS algorithm, cancel the integer conversion
            StoredItems = knapsack_FPTAS(items, Storage) # knapsack_FPTAS, knapsack01_dp
            
            # Replicated contents are recorded in StoredItems
            for item in StoredItems:
                i = int(item[0])
                x.add(i)
                # Increase bandwidth saved by replication
                if t[i] >= 0:
                    BW[t[i]] += RequestNumOfID[i] * Size[i]                
                t[i] = -1                
                # After setting a content, delete it from other neighbors
                for s in T_avail:
                    s.discard(i)
            '''
            # The M-th neighbor is abstract of local replication, and its bottleneck is on the storage capacity            
            for i in kContentAvail:
                if ReqSize[i] > 0 and Storage - Size[ContentAvailList[i]] >= 0:
                    x.add(ContentAvailList[i])
                    t[ContentAvailList[i]] = -1
                    Storage -= Size[ContentAvailList[i]]
                    # Note, after setting a content, delete it from other neighbors
                    for s in T_avail:
                        s.discard(ContentAvailList[i])
            '''
        else:                
            # The bottleneck of other neighbors is bandwidth
            for i in kContentAvail:
                if ReqSize[i] > 0 and BW[j] - ReqSize[i] >= 0:                
                    t[ContentAvailList[i]] = j
                    BW[j] -= ReqSize[i]
                    # Note, after setting a content, delete it from other neighbors
                    for s in T_avail:
                        s.discard(ContentAvailList[i]) 
    # Calculate profit
    profit = 0
    for i in range(N):
        if t[i] >= 0:
            profit += RequestNumOfID[i]*Size[i]*P[t[i]]
    # Calculate bandwidth usage
    #averageUtiBW = 1 - sum(BW)/(M*4e4)
    [ProviderList, CustomerList, PeerList] = GetNeighbors(ASN)
    # Calculate bandwidth usage of providers. For Tier-1, calculate its peers.
    ProviderNum = len(ProviderList)
    PeerNum = len(PeerList)
    
    if ProviderNum > 0:        
        averageUtiBW_P = 1-sum(BW[:ProviderNum])/(ProviderNum*4e4)
    elif PeerNum > 0:
        averageUtiBW_P = 1-sum(BW[PeerNum:])/(PeerNum*4e4)
    else:
        print 'No Providers and Peers'    
   
    # Record the list of contents which cannot be obtained from providers due to the limit of their bandwidth
    NotContentList = []
    for i in range(N):
        if RequestNumOfID[i] > 0:
#            print 'Requesting content %d for %d times.'%(i, RequestNumOfID[i])
#            print 'Content %d can be get from neighbor %d'%(i, t[i])            
#            print 'Price of neighbor %d is %d'%(t[i], P[t[i]])
            if t[i] == -2:
                NotContentList.append(i)
                
    if len(NotContentList) > 0:
        print 'Number of Contents not found in Algo3 is: %d'%len(NotContentList)
        print 'Remain BW of the 10 providers of Algo3 are:'
        print BW[:10]
  
           
    return profit, averageUtiBW_P

##############################Algorithm 4################################
# Output the profit and bandwidth usage
def algo4(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size):
    # Deep copy the variables which will be changed in this function    

    BW = list(BW)
    BWOrigin = BW[:]
    
    T_avail = list(copy.deepcopy(T_avail))
    BW = dict(zip(range(M), BW))
    Tavail = dict()
    for i in range(N):
            for j in range(M):
                    if i in T_avail[j]:
                            Tavail[(i,j)] = 1
                    else:
                            Tavail[(i,j)] = 0
    #for i in range(N):
    #    print Tavail[(i, 10)],

    RequestNumOfID = dict(zip(range(N), RequestNumOfID))
    Size = dict(zip(range(N), Size))
    P = dict(zip(range(M), P))

    prob = LpProblem("Algo4", LpMaximize)
    x = LpVariable.dicts("x", range(N), 0, 1, LpBinary)
    t = LpVariable.dicts("t", [(i, j) for i in range(N) for j in range(M)], 0, 1, LpInteger)

    # Define the goal function
    prob += lpSum(RequestNumOfID[i]*Size[i]*P[j]*t[(i,j)] for i in range(N) for j in range(M))
    prob += lpSum(Size[i] * x[i] for i in range(N)) <= Storage

    for j in range(M):
        prob += lpSum(RequestNumOfID[i]*Size[i]*t[(i,j)] for i in range(N)) <= BW[j]

    for i in range(N):
        for j in range(M):
            prob += t[(i,j)] <= Tavail[(i,j)]
            
    for i in range(N):        
        prob += x[i] + lpSum(t[(i,j)] for j in range(M)) == 1        

    #prob.writeLP("Algo4.lp")
    prob.solve(GUROBI())
    print "Status:", LpStatus[prob.status]

    # Return the results
    tFinal = [-2] * N
    xFinal = set([])
    for i in range(N):
        if x[i].varValue > 0:            
            xFinal.add(i)
    for i in range(N):
        for j in range(M):        
            if t[(i,j)].varValue > 0 and RequestNumOfID[i] > 0:
                tFinal[i] = j
                #print 't[(%d, %d)=%d'%(i,j ,t[(i,j)].varValue)

    #print 'x=', xFinal
    #print 't=', tFinal

    # Calculate profit
##    profit = 0
##    for i in range(N):
##        if tFinal[i] >= 0:
##            profit += RequestNumOfID[i]*Size[i]*P[tFinal[i]]
##    print 'profit compute', profit
    # Calculate remained bandwidth
    # Calculate bandwidth usage
    [ProviderList, CustomerList, PeerList] = GetNeighbors(ASN)
    # Calculate the bandwidth of providers, if it is a Tier-1 AS, calculate its peers
    ProviderNum = len(ProviderList)
    PeerNum = len(PeerList)
    
    
    if ProviderNum ==0:
        print 'No Providers'
        return
        
    #print 'ProviderNum', ProviderNum
    remainBW = BWOrigin    
    for i in range(N):        
        if tFinal[i] >= 0:
            remainBW[tFinal[i]] -= RequestNumOfID[i]*Size[i]
            
    averageUtiBW_P = 1-sum(remainBW[:ProviderNum])/(ProviderNum*4e4)

    #print 'Profit of Algo4:', prob.objective.value()
    return prob.objective.value(), averageUtiBW_P
    

##########################################################################
def main(ReqNum, alpha=0.7, ratioStorage=0.2):
        
    ASListCare = [1239, 3356, 7018, 8002, 25973, 5400, 14744, 3209, 7713]
    ASN = 5400
    [ProviderList, CustomerList, PeerList] = GetNeighbors(ASN)
    #print 'Provider number: %d'%len(ProviderList)
    #print 'Customer number: %d'%len(CustomerList)
    #print 'Peer number: %d'%len(PeerList)

    # All the IP prefixes of ASes we care for have been preprocessed and saved in the list structures
    f = open('./MyData/Pickle/as2prefix/NeighborPrefixesAS%d.txt'%ASN, 'r')
    PrefixListSelf = pickle.load(f)
    PrefixListProvider = pickle.load(f)
    PrefixListCustomer = pickle.load(f)
    PrefixListPeer = pickle.load(f)
    f.close()

    PrefixNumberSelf = map(len, PrefixListSelf)
    PrefixNumberProvider = map(len, PrefixListProvider)
    PrefixNumberCustomer = map(len, PrefixListCustomer)
    PrefixNumberPeer = map(len, PrefixListPeer)

    # Fixed bus: one of the customer has zero IP prefixes, and it makes the zipf function loops forever
    # Delete ASes whoes length is 0
    PrefixNumberSelf = [i for i in PrefixNumberSelf if i >0]
    PrefixNumberProvider = [i for i in PrefixNumberProvider if i >0]
    PrefixNumberCustomer = [i for i in PrefixNumberCustomer if i >0]
    PrefixNumberPeer = [i for i in PrefixNumberPeer if i >0]

    assert (0 not in PrefixNumberSelf and 0 not in PrefixNumberProvider and
            0 not in PrefixNumberCustomer and 0 not in PrefixNumberPeer)


##################################################################################
    #added for small space for algo4
##    PrefixNumberCustomer = [i/30 for i in PrefixNumberCustomer]
##    PrefixNumberPeer = [i/30 for i in PrefixNumberPeer]
##
##    PrefixNumberSelf = [i for i in PrefixNumberSelf if i >0]
##    PrefixNumberProvider = [i for i in PrefixNumberProvider if i >0]
##    PrefixNumberCustomer = [i for i in PrefixNumberCustomer if i >0]
##    PrefixNumberPeer = [i for i in PrefixNumberPeer if i >0]
##
##    print 'PrefixNumberCustomer', PrefixNumberCustomer
##    print 'PrefixNumberPeer', PrefixNumberPeer


###################################################################################

    ContentNum = sum(PrefixNumberCustomer) + sum(PrefixNumberPeer)    
    # ContentNum is calculated according to ASN
    # Set constants
    # N is number of contents
    ProviderNum = len(PrefixNumberProvider)
    CustomerNum = len(PrefixNumberCustomer)
    PeerNum = len(PrefixNumberPeer)
    N = ContentNum
    # M is number of neighbors
    M = ProviderNum + CustomerNum + PeerNum
    print 'N=',N
    print 'M=',M

    #print '---------------------------------------------------'
    #print 'Content number: %d, neighbor number: %d'%(N, M)
    beta = 1   # Bandwidth usage
    # Suppose the unit of content size if MB, and the bandwidth between neighbors is 40Gbps
    BW_Provider = [4e4*beta] * ProviderNum
    BW_Customer = [4e4*beta] * CustomerNum
    BW_Peer = [4e4*beta] * PeerNum

    #BW_All = sum(BW_Provider) + sum(BW_Customer) + sum(BW_Peer)
    #print 'BW_All=%d'%BW_All


    ##############################Constants setting####################################
    # Zipf parameter of content ID
    alphaID = 0.7
    # According to the paper of "Web zipf evidence", alpha is between 0.64 and 0.83
    # The values of alpha for the six traces are shown in Table I
    alphaRequest = alpha  #float(sys.argv[2])
    # Set the seed
    #random.seed(10)

    # Number of requests
    #RequestNum = int(BW_All/1.7*0.9 * 0.13)
    RequestNum = int(ReqNum * 1e3)  #int(BW_All/1.7*0.9* ReqNum)   #int(N * ReqNum)    #15-40(Size=1)
    #Storage = int(N * ratioStorage * 2) # It needs to be a integer when calling dynamic programming. Its average size is 1.7M
    # The size of contents conforms to normal distribution, and the expected value is 17K
    Size = [] #[1] * N #[]
    for i in range(N):
        while True:
            vPareto = 3.2*random.paretovariate(1.3) # Make sure the expected value is 1.7
            if vPareto < 20:    # Only consider values less than 20M
                Size.append(vPareto)
                break        
        #Pareto distribution for the tail (with alpha, 1-1.5, we choose 1.3)
        #random.gauss(1.7, 0.3)) # The Gauss distribution

    Storage = int(sum(Size) * ratioStorage) + 1 # Integer conversion of dynamic programming, and the average size is 1.7M
    # Sort according to the number of IP prefixes of providers
    #print PrefixNumberProvider
    # Generate some price values randomly, and sort from small to large
    Price_Provider = [0] * ProviderNum
    temp = []
    for i in range(ProviderNum):
        temp.append(random.uniform(-2, -1))
    temp.sort()
    #print temp
    # Sort the values from small to large, and save the corresponding subscripts in the list
    # Suppose the more IP prefixes an ISP has, the higher its price, since its has a larger scale and better connectivity
    temp2 = (-np.array(PrefixNumberProvider)).argsort()
    for i in range(ProviderNum):
        Price_Provider[temp2[i]] = temp[i]
    #print Price_Provider

    # Calculate the prices of customers, and the smaller its size, the higher it needs to pay
    Price_Customer = [0] * CustomerNum
    temp = []
    for i in range(CustomerNum):
        temp.append(random.uniform(1, 2))
    temp.sort()

    # Suppose ISPs with more IP prefixes pay lower price to providers
    temp2 = (-np.array(PrefixNumberCustomer)).argsort()
    for i in range(CustomerNum):
        Price_Customer[temp2[i]] = temp[i]
    del temp
    del temp2

    # The prices of peers
    Price_Peer = [0] * PeerNum

    # Generate the requests list
    RequestNumOfID = [0] * N
    zfRequest = ZipfGenerator(N, alphaRequest)
    for i in range(RequestNum):
        RequestNumOfID[zfRequest.next()] += 1

    # Calculate reachability, i.e. T_avail
    # Record the content ID of customers and peers
    IDListCustomer = []
    IDListPeer = []
    # Generate content ID by zipf for every neighbor AS to make more common IDs
    zfID = ZipfGenerator(N, alphaID)

    #for i in PrefixNumberCustomer:
    #    IDListCustomer.append(set(random.sample(ContentIDList, i)))
    #for i in PrefixNumberPeer:
    #    IDListPeer.append(set(random.sample(ContentIDList, i)))        

    # Generate the content set of Customers and Peers
    for i in PrefixNumberCustomer:
        zSet = set([])           
        while True:        
            zSet |= set([zfID.next()])
            if len(zSet) == i:
                IDListCustomer.append(zSet)
                break
          
    for i in PrefixNumberPeer:
        zSet = set([])           
        while True:        
            zSet |= set([zfID.next()])
            if len(zSet) == i:
                IDListPeer.append(zSet)
                break

    assert map(len, IDListCustomer) == PrefixNumberCustomer
    assert map(len, IDListPeer) == PrefixNumberPeer

        
    #print 'The final customer content number is: %d'%len(IDListCustomer)
    #print 'The final peer content number is: %d'%len(IDListPeer)

    # Combine the price, bandwidth, and reachability of providers, customers, and peers
    # The order of neighbors is accordance with providers, customers, and peers
    P = []
    P.extend(Price_Provider)
    P.extend(Price_Customer)
    P.extend(Price_Peer)

    BW = []
    BW.extend(BW_Provider)
    BW.extend(BW_Customer)
    BW.extend(BW_Peer)


    IDListProvider = [set(range(N))] * ProviderNum
    T_avail = []
    T_avail.extend(IDListProvider)
    T_avail.extend(IDListCustomer)
    T_avail.extend(IDListPeer)

    assert len(T_avail) == M
    assert len(P) == M
    assert len(BW) == M

    # Converse the constants to tupple type in case it will be changed in functions (actually, it needs deep copy, since tuple will still be changed)
    # N, M, Storage is native type, and will not be changed in functions
    P = tuple(P)
    BW = tuple(BW)
    T_avail = tuple(T_avail)
    RequestNumOfID = tuple(RequestNumOfID)
    Size = tuple(Size)

 
    # Run the three different algorithms with the same parameters
    print '----------runtime statistics---------------------------'
    print 'Beginning Algo1'
    print time.asctime() 
    beginofAlgo1 = time.time()
    profit1, averageUtiBW_P1 = algo1(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    endofAlgo1 = time.time()
    t1 = endofAlgo1 - beginofAlgo1
    print 'Execute time of Algo1 is: %ds'%t1    
    print 'Beginning Algo2'
    print time.asctime()
    profit2, averageUtiBW_P2 = algo2(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    beginofAlgo3 = time.time()
    t2 = beginofAlgo3 - endofAlgo1
    print 'Execute time of Algo2 is: %ds'%t2 
    print 'Beginning Algo3'
    print time.asctime()

    profit3, averageUtiBW_P3 = algo3(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    endofAlgo3 = time.time()    
    t3 = endofAlgo3 - beginofAlgo3
    print 'Execute time of Algo3 is: %ds'%t3
    print 'Beginning Algo4'
    print time.asctime()

    profit4, averageUtiBW_P4 = 0, 0 #algo4(ASN, N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    endofAlgo4 = time.time()
    t4 = endofAlgo4 - endofAlgo3
    print 'Execute time of Algo4 is: %ds'%t4
    print 'End of Algo4'
    print time.asctime()
    print '--------------------------------------------------------'


    return (int(profit1), averageUtiBW_P1, int(profit2), averageUtiBW_P2,
            int(profit3), averageUtiBW_P3, int(profit4), averageUtiBW_P4,
            t1, t2, t3, t4)

    '''
    if algo == 1:
        return algo1(N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    elif algo == 2:
        return algo2(N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    elif algo == 3:
        return algo3(N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    elif algo == 4:
        return algo4(N, M, Storage, P, BW, T_avail, RequestNumOfID, Size)
    else:
        print 'algo error',algo
        return 0
    '''
############################################################################
c1 = []
c2 = []
c3 = []
c4 = []

u1=[]
u2=[]
u3=[]
u4=[]

t = []

print 'Beginning time:', time.asctime()
'''
#############################Experiment 1#######################################
for n in range(5):
    for i in arange(10, 180, 10):
        (profit1, averageUtiBW_P1, profit2, averageUtiBW_P2,
         profit3, averageUtiBW_P3, profit4, averageUtiBW_P4,
         t1, t2, t3, t4) = main(i)
        print 'profit1=%d'%profit1
        print 'averageUtiBW_P1=%.4f'%averageUtiBW_P1
        print 'profit2=%d'%profit2
        print 'averageUtiBW_P2=%.4f'%averageUtiBW_P2
        print 'profit3=%d'%profit3
        print 'averageUtiBW_P3=%.4f'%averageUtiBW_P3
        print 'profit4=%d'%profit4
        print 'averageUtiBW_P4=%.4f'%averageUtiBW_P4
        c1.append(profit1)
        c2.append(profit2)
        c3.append(profit3)
        c4.append(profit4)

        u1.append(averageUtiBW_P1)
        u2.append(averageUtiBW_P2)
        u3.append(averageUtiBW_P3)
        u4.append(averageUtiBW_P4)

        t.append([t1, t2, t3, t4])
        
    print 'c1=', c1
    print 'c2=', c2
    print 'c3=', c3
    #print 'c4=', c4
    print 'u1=', u1
    print 'u2=', u2
    print 'u3=', u3
    #print 'u4=', u4
    #print 't=', t

    print 'End time: ', time.asctime()
    fileName = 'RequestsAS7719-%d'%n
    f = open(fileName, 'w')
    pickle.dump(c1, f)
    pickle.dump(c2, f)
    pickle.dump(c3, f)
    #pickle.dump(c4, f)
    

    pickle.dump(u1, f)
    pickle.dump(u2, f)
    pickle.dump(u3, f)
    #pickle.dump(u4, f)

    #pickle.dump(t, f)

    f.close()  


#############################Experiment 2#######################################
for n in range(5):
    for i in arange(0.6, 1.35, 0.05):
        (profit1, averageUtiBW_P1, profit2, averageUtiBW_P2,
         profit3, averageUtiBW_P3, profit4, averageUtiBW_P4,
         t1, t2, t3, t4) = main(120, i)
        print 'profit1=%d'%profit1
        print 'averageUtiBW_P1=%.4f'%averageUtiBW_P1
        print 'profit2=%d'%profit2
        print 'averageUtiBW_P2=%.4f'%averageUtiBW_P2
        print 'profit3=%d'%profit3
        print 'averageUtiBW_P3=%.4f'%averageUtiBW_P3
        print 'profit4=%d'%profit4
        print 'averageUtiBW_P4=%.4f'%averageUtiBW_P4
        c1.append(profit1)
        c2.append(profit2)
        c3.append(profit3)
        #c4.append(profit4)

        u1.append(averageUtiBW_P1)
        u2.append(averageUtiBW_P2)
        u3.append(averageUtiBW_P3)
        #u4.append(averageUtiBW_P4)

        t.append([t1, t2, t3, t4])
        
    print 'c1=', c1
    print 'c2=', c2
    print 'c3=', c3
#    print 'c4=', c4
    print 'u1=', u1
    print 'u2=', u2
    print 'u3=', u3
#    print 'u4=', u4
#    print 't=', t

    print 'End time: ', time.asctime()
    fileName = 'ZipfAS7719-%d'%n
    f = open(fileName, 'w')
    pickle.dump(c1, f)
    pickle.dump(c2, f)
    pickle.dump(c3, f)
    pickle.dump(c4, f)
    

    pickle.dump(u1, f)
    pickle.dump(u2, f)
    pickle.dump(u3, f)
    pickle.dump(u4, f)

    pickle.dump(t, f)

    f.close()  

'''
#############################Experiment 3########################################
for n in range(2):
    for i in arange(0.1, 1, 0.1):
        (profit1, averageUtiBW_P1, profit2, averageUtiBW_P2,
         profit3, averageUtiBW_P3, profit4, averageUtiBW_P4,
         t1, t2, t3, t4) = main(90, 0.7, i)
        print 'profit1=%d'%profit1
        print 'averageUtiBW_P1=%.4f'%averageUtiBW_P1
        print 'profit2=%d'%profit2
        print 'averageUtiBW_P2=%.4f'%averageUtiBW_P2
        print 'profit3=%d'%profit3
        print 'averageUtiBW_P3=%.4f'%averageUtiBW_P3
        print 'profit4=%d'%profit4
        print 'averageUtiBW_P4=%.4f'%averageUtiBW_P4
        c1.append(profit1)
        c2.append(profit2)
        c3.append(profit3)
#        c4.append(profit4)

        u1.append(averageUtiBW_P1)
        u2.append(averageUtiBW_P2)
        u3.append(averageUtiBW_P3)
#        u4.append(averageUtiBW_P4)

        t.append([t1, t2, t3, t4])
        
    print 'c1=', c1
    print 'c2=', c2
    print 'c3=', c3
#   print 'c4=', c4
    print 'u1=', u1
    print 'u2=', u2
    print 'u3=', u3
#    print 'u4=', u4
#    print 't=', t

    print 'End time: ', time.asctime()
    fileName = 'StorageAS5400-%d'%n
    f = open(fileName, 'w')
    pickle.dump(c1, f)
    pickle.dump(c2, f)
    pickle.dump(c3, f)
    pickle.dump(c4, f)
    

    pickle.dump(u1, f)
    pickle.dump(u2, f)
    pickle.dump(u3, f)
    pickle.dump(u4, f)

    pickle.dump(t, f)

    f.close()  

