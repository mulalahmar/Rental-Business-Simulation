
# coding: utf-8

# # Discrete-Event Simulation

# In[1]:


import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)

import numpy as np
import string
import random
import copy
import operator
import itertools
from scipy.stats import rv_discrete 


# In[43]:

#- Variables and Parameters Initilization
#------------------------------------------------------------------------

# Systematically generated parameters
automatic = False

# Initialize parameters

n_iter = 5    # number of iterations
n_cust = 3    # number of servers
n_class = 6   # number of inventory classes
min_res = 3   # minimum number of items to be reserved at any time
max_cap = 2   # Maximum number of units in inventory of each class
custs = range(n_cust)
items = range(n_class)


# Initial Counter Variables

stockout_count = np.zeros(n_cust)
tot_stockout_count = 0

# Initial State of System

ser_state = np.zeros((n_cust, n_class))
ser_state = np.array([[0,1,1,0,1,0],[1,1,1,0,0,0],[0,1,0,0,1,1]])

res_state = np.ones((n_cust, n_class))
res_state = np.array([[1,1,0,0,0,0],[1,0,0,0,1,0],[0,1,0,0,1,0]])

max_inv = np.array(map(lambda x: int(x),(np.ones(n_class)*max_cap)))
inv_state = np.array(map(lambda x: int(x),(np.zeros(n_class))))
invpos_state = np.array(map(lambda x: int(x),(np.zeros(n_class))))


# Transition probabilities
Mu = np.ones((n_cust,n_class))               # Rate of service of items
Lambda = np.ones((n_cust,n_class))*1         # Rate of deletion of items from closet
Nu = np.ones((n_cust,n_class))*1.5           # Rate of addition of items to closet


# Dispatching Decision Heuristic 

#disp_policy = 'Random'
#disp_policy = 'HighestInv'
disp_policy = 'HighestInvPostSignal'


# Probablities

#if automatic:
#    p = [random.random() for i in range(n_class)]
#    ones = np.ones(n_class)
#    probs = np.around(adj_prob(p,ones), decimals=4)
#else:
#    probs = [0.2, 0.1, 0.3, 0.1, 0.2, 0.1]


#- Functions
#------------------------------------------------------------------------

def random_select(values,probabilities,count):  
    distrib = rv_discrete(values=(values, probabilities))
    value = int(distrib.rvs(size=count))
    return value
    
def multipl_list(list1,list2):
    # This function multiplies elements of two vectors and outpus them in a vector
    return map(lambda x: x[0]*x[1], zip(list1, list2))

def dotprod_list(list1,list2):
    # This function multiplies elements of two vectors and outpus the sum of the outputs
    return sum(i[0] * i[1] for i in zip(list1, list2))

def adj_prob(list1,list2):
    # This function adjusts probabilities based on feasible transitions
    tot = dotprod_list(list1,list2)
    mult = multipl_list(list1,list2)
    return map(lambda x: x/tot, mult)

def feas_trans(state1,state2):
    # This function identifies feasible states for a transition and outputs a vector
    return map(lambda x: max(x[0]-x[1],0), zip(state1, state2))

def compl_state(state):
    # This function identifies feasible states for a transition and outputs a vector
    state_ones = np.ones((len(state),), dtype=np.int)
    return map(lambda x: x[0]-x[1], zip(state_ones, state))

def sum_statevalues(state):
    # This function adds elements across all lists within the sublist
    return ([sum(sublist) for sublist in itertools.izip(*state)])

def update_inv():
    inv_state = max_inv - (sum_statevalues(ser_state))
    invpos_state = inv_state  - (sum_statevalues(res_state))
    return inv_state, invpos_state

def dec_rule(inv_state, r_state,s_state,rule):
    
    # This function applies the pre-selected heuristic decision rule 
    # using current system state information to dispatch an item to 
    # a customer based on the set of items they selected 
    
    if rule == 'HighestInvPostSignal':
        
        # This decision rule selects an item among the customer 
        # selected items with the highest inventory on-hand level 
        # minus all signaled demand while inventory on-hand is positive. 
        # If multiple items have the same inventory on-hand levels, 
        # it breaks the tie randomly. 
 
        inv_state, invpos_state = update_inv()
        state = feas_trans(r_state,s_state)
        print state
        
        if sum([1 for i in range(len(inv_state)) if (inv_state[i]>0) & (state[i]==1)])>0: 
            #print "-------------------Some of desired items have inventory"
            release(s_state,item_n) 
            inv_state, invpos_state = update_inv()
            print invpos_state, inv_state, r_state,s_state

            sel_state = np.zeros(n_class)
            for i in range(len(inv_state)):
                if (inv_state[i]>0) & (state[i]==1): 
                    sel_state[i]=1
                else:
                    sel_state[i] = np.nan
            # select from reserved items with the highest inventory position items
            inv_state_adj = np.array(multipl_list(invpos_state,sel_state))
            #print inv_state_adj, invpos_state, sel_state
        
            max_index = np.where(inv_state_adj==np.nanmax(inv_state_adj))[0]
            #print 'max_index', max_index
            index_select = int(random.random()*(len(max_index)))
            #print 'index_select', index_select
            item_select = max_index[index_select]
            #print 'item_select', item_select
            print '** cust received item ', item_select+1
            #Item removed from selection and added to service
            r_state[item_select] = r_state[item_select] - 1
            s_state[item_select] = s_state[item_select] + 1 

        else:
            global stockout_count
            #print "-------------------None of desired items is in inventory"
            print '!! Stock out !!'
            stockout_count[cust_n] +=1

                
    if rule == 'HighestInv':
        
        # This decision rule selects the item with the highest 
        # inventory on-hand level among the customer selected items. 
        # If multiple items have the same inventory on-hand levels, 
        # it breaks the tie randomly.
        
        inv_state, invpos_state = update_inv()
        state = feas_trans(r_state,s_state)
        
        if sum([1 for i in range(len(inv_state)) if (inv_state[i]>0) & (state[i]==1)])==0: 
            #print "-------------------All desired items have no inventory"
            sel_state = state
            for i in range(len(sel_state)):
                if (sel_state[i]==0): sel_state[i] = np.nan
            inv_state_adj = np.array(multipl_list(inv_state,sel_state))
        else:
            #print "-------------------Some of desired items have inventory"
            sel_state = np.zeros(n_class)
            for i in range(len(inv_state)):
                if (inv_state[i]>0) & (state[i]==1): 
                    sel_state[i]=1
                else:
                    sel_state[i] = np.nan
            # select from reserved items with the highest inventory position items
            inv_state_adj = np.array(multipl_list(invpos_state,sel_state))          
        
        max_index = np.where(inv_state_adj==np.nanmax(inv_state_adj))[0]
        #print 'max_index', max_index
        index_select = int(random.random()*(len(max_index)))
        #print 'index_select', index_select
        item_select = max_index[index_select]
        #print 'item_select', item_select
        print '** cust received item ', item_select+1
        
        #Item removed from selection and added to service
        r_state[item_select] = r_state[item_select] - 1
        s_state[item_select] = s_state[item_select] + 1 

    if rule == 'Random':
        
        # This decision rule selects one item randomly among the 
        # customer selected items. The selection process follows
        # a discrete uniform distribution.
        
        inv_state, invpos_state = update_inv()
        state = feas_trans(r_state,s_state)
        
        #Random probability among reserved items
        state_ones = map(lambda x: float(x/len(state)), np.ones(len(state)))
        probs_adj = adj_prob(state,state_ones)
        item_select = random_select(items,probs_adj,1)
        print '** cust received item ', item_select+1
        
        #Item removed from selection and added to service
        r_state[item_select] = r_state[item_select] - 1
        s_state[item_select] = s_state[item_select] + 1 

def reserve(r_state,item_select):
    
    # This function applies the customer item choice distribution 
    # to the system state by selecting an item available in inventory
    # and add it to the customer closet

    #Item reserved from inventory
    r_state[item_select] = r_state[item_select]+1
    print '++ cust reserved item ', item_select+1
    
def delete(r_state,item_select):
    
    # This function applies the customer item choice distribution 
    # to the system state by deleting a pre-selected item from 
    # the customer closet

    #Item reserved from inventory
    r_state[item_select] = r_state[item_select]-1
    print '-- cust deleted item ', item_select+1
    
def release(s_state,item_select):
    
    # This function applies the customer service (usage) time 
    # distribution to the system state to release one item from 
    # current items in service and add it to inventory

    #Item removed from service
    s_state[item_select] = s_state[item_select]-1
    print '- cust released item ', item_select+1

def main():

    #- Main
    #------------------------------------------------------------------------

    # Simulation runs multiple iterations
    print 'Initial System State'
    print 'Service'
    print ser_state
    print 'Signals'
    print res_state

    n_iter = 10#00

    for l in range(n_iter):

        # Customer can add as many items to closet as desired
        add_rate = ((1-res_state)*Nu)

        # Customer can delete items from closet but should keep min_res items in closet
        res_state_adj = copy.deepcopy(res_state)
        res_state_adj[np.where(np.sum(res_state_adj,axis=1)<=min_res),0:n_class]=np.zeros(n_class,int)
        del_rate = (res_state_adj*Lambda)

        # Customer can return items from home but will not receive items if (technically block return)
        # -its closet has below min_res items
        # -all items in closet are out of stock    
        ser_state_adj = copy.deepcopy(ser_state)
        ####### To be tested
        ser_state_adj[np.where(np.sum(res_state_adj,axis=1)<=min_res),0:n_class]=np.zeros(n_class,int)   
        ####### To be tested
        rel_rate = (ser_state_adj*Mu)


        trans_rate = np.concatenate((np.concatenate((rel_rate, add_rate), axis=0),del_rate), axis=0)
        #print trans_prob
        #tot = sum(sum((trans_rate)))
        #print tot
        trans_prob = trans_rate/sum(sum((trans_rate)))
        print trans_prob
        #print trans_prob.shape

        select_cell = random_select(range(0,n_cust*n_class*3),trans_prob.flatten(),1)
        #print 'selection', select_cell, n_cust, n_class
        #row_n = (select_cell/ n_class)
        #print 'row', select_cell/ n_class
        #col_n = (select_cell% n_class)
        #print 'col',(select_cell% n_class)
        trans_n = (select_cell/ (n_class*n_cust))
        #print 'trans',(select_cell/ (n_class*n_cust))
        #print '(Cust=',((row_n)%3),',Item=',col_n,',Transition=',trans_n,')'
        cust_n = ((select_cell/ n_class)%3)
        item_n = (select_cell% n_class)

        # trans_n =
        # 0 for release
        # 1 for reserve    
        # 2 for delete

        if trans_n == 0:  # 0 for release

            print '\n------ Cust ',str(cust_n+1),' releases item', str(item_n+1)
            inv_state, invpos_state = update_inv()
            print invpos_state, inv_state, res_state[cust_n],ser_state[cust_n]

            #Decision to release item, remove one from closet and add item to service
            dec_rule(inv_state,res_state[cust_n],ser_state[cust_n],disp_policy)
            inv_state, invpos_state = update_inv()
            print invpos_state, inv_state, res_state[cust_n],ser_state[cust_n]

        elif trans_n == 1:  # 1 for reserve         

            print '\n------ Cust ',str(cust_n+1),' reserves item', str(item_n+1)
            inv_state, invpos_state = update_inv()
            print invpos_state, inv_state, res_state[cust_n],ser_state[cust_n]
            #Item item_n added to closet cust_n
            reserve(res_state[cust_n],item_n) 
            inv_state, invpos_state = update_inv()
            print invpos_state, inv_state, res_state[cust_n],ser_state[cust_n]

        else:              # 2 for delete

            print '\n------ Cust ',str(cust_n+1),' deletes item', str(item_n+1)
            inv_state, invpos_state = update_inv()
            print invpos_state, inv_state, res_state[cust_n],ser_state[cust_n]
            #Item item_n deleted from closet cust_n
            delete(res_state[cust_n],item_n) 
            inv_state, invpos_state = update_inv()
            print invpos_state, inv_state, res_state[cust_n],ser_state[cust_n]

    print '--------- Stock-out Counts'
    print 'Probability of stock-outs =', sum(stockout_count)/n_iter
    print 'Stock-outs by Customers', stockout_count  

if __name__ == '__main__':
    main()
    

