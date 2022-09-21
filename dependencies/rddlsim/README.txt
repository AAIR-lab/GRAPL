Content from README.md
======================

RDDLSim -- A simulator for the relational dynamic influence diagram language (RDDL).

Implements a parser, simulator, and client/server evaluation architecture for the relational dynamic influence diagram language (RDDL) -- pronounced "riddle". RDDL is intended to compactly support the representation of a wide range of relational MDPs and POMDPs and support the efficient simulation of these domains. RDDL is used in a number of past and present International Probabilistic Planning Competitions (IPPCs):

[IPPC 2011 (Discrete)](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/)
[IPPC 2014 (Discrete)](https://cs.uwaterloo.ca/~mgrzes/IPPC_2014/)
[IPPC 2015 (Continuous)](http://users.cecs.anu.edu.au/~ssanner/IPPC_2014/)

RDDL Resources:

[RDDL Language Guide](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf)

Please cite as

```
   @unpublished{Sanner:RDDL,
      author = "Scott Sanner",
      title = "Relational Dynamic Influence Diagram Language (RDDL): Language Description",
      note = "http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf",
      year = 2010}
```

[RDDL Tutorial](http://users.rsise.anu.edu.au/~ssanner/Papers/RDDL_Tutorial_ICAPS_2014.pdf)

When you checkout the code, the first file you'll want to look at is [INSTALL.txt](https://github.com/ssanner/rddlsim/blob/master/INSTALL.txt)... this includes everything you need to start simulating, visualizing, and translating domains in < 5 minutes!

If you want to use an MDP or POMDP planner in conjunction with RDDL, please check out planner releases at the above competition web pages... all support the RDDL Client/Server protocol.


Additional Information from Original README.txt
===============================================

LICENSE.txt:  GPLv3 license information for RDDLSim source and alternate
              license information for redistibuted 3rd party software

INSTALL.txt:  RDDLSim installation and execution instructions

PROTOCOL.txt: RDDLSim client/server protocol
