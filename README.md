## Copyright
Written by Zhen Feng of Tsinghua University
Copyright GPLv2
The code is written for the simulation experiment of my research papers.
Dataset is from CAIDA

## 1. Datasets
The original dataset is located in the 'OriginData' directory. 20131101.as-rel.txt is about the relationship between ASes in the Internet, and routeviews-rv2-20140401-1400.pfx2as is about the IP prefixes and which AS they belong. 

## 2. Programs
### (1) neighbors.py
Record neighbor ASes of the ASes we care for, and the results are located in derectory './MyData/Neihbors/'. The file name indicates the AS number, and the data include three lines of AS number, i.e., provider, customer, and peer.
### (2)Prefix.py
Record the AS number in routeviews.pfx2as, and its own IP prefixes.
The results are lcoated in directory './MyData/Pickle/AsList.txt' and './MyData/Pickle/PrefixList.txt'
### (3)Fun.py
Including two functions
GetNeighbors(asn) will return three lists of an AS, they include provider list, customer list and peer list.
GetPrefix(asnList) will return a IP prefixes list of an AS
### (4)ASN2Prefix.py
Save the IP prefixes of the neighbor ASes of the 9 ASes we care for.
The results are save in pickle files, and the format for an AS is like 'NeighborPrefixesAS%d.txt'%ASN.
### (5)CheckPrefixOverlap.py
Find out the IP prefixes from customer and peer ASes, and conbine them and exclude the overlapping we will get the number of all the IP prefixes. We treat it as the number of all the contents in the simulation. The contents of every customer or peer AS can be in accordance with its IP prefixes.
Note, this program is obsolete since most overlapping only occurs not between ASes, but in an AS.
### (6) AlgoMain.py
The main program of simulation.
Generate different distributions and calculate different metrics.

