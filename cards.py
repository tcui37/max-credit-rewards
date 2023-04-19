import csv
import torch
import pandas as pd
import numpy as np
from itertools import combinations

class CardMatrix:
    def __init__(self,data_path: str):

        # np array data
        self.data = pd.read_csv(data_path,header=None).to_numpy()
        self.m,self.n = self.data.shape


        #tensor generation, strip labels
        a = np.vstack([self.data[2:,2:-1]]).astype(np.float16)
        self.tensor = torch.from_numpy(a)

        # create indexing for indexing into tensor
        m,n = self.data.shape
        self.card2index = {self.data[i][0]:i-2 for i in range(2,m)} 
        self.index2card = self.data[2:,0].T
        self.category2index = {self.data[0][i]:i-2 for i in range(2,n-1)}
        self.index2category = self.data[0,2:].T


        # Manual edits to self.tensor

        # spread 'all' reward across all empty values
        for card_row in self.tensor[:,2:]:
            card_row[card_row == 0 ] = card_row[-2]

        # manualy edit tensor to be -.03 across all rent except for BILT (couldnt figure out how to parallelize this ;-;)
        bilt_index = self.card2index['BILT']
        for row in range(self.tensor.size()[0]):
            if row != bilt_index:
                self.tensor[row,17] = self.tensor[row,17]  - 0.03



    def __repr__(self) -> str:
        rep = ''
        for row in self.data:
            rep += '\n' + str(row)
        return rep
    
    def get_tensor(self): return self.tensor
    
    def get_card_indices(self): return self.card2index

    def get_category_indices(self): return self.category2index

    def eval_cards(self,spendings:dict,ids: list) -> float:
        m = len(self.category2index)
        spending_tensor = torch.zeros(m,dtype=torch.float32)
        for category,spent in spendings.items():
            spending_tensor[self.category2index[category]] = float(spent)

        indices = [self.card2index[card_id] for card_id in ids ]
        #
        submatrix = self.tensor[indices]

        # sum fees, sum credits, select maximal credit across cards
        fees = submatrix[:,0].sum(dim=0,keepdims=True)
        credits = submatrix[:,1].sum(dim=0,keepdims=True)
        max_categories = submatrix.max(dim=0,keepdims=False)
        max_categories_rewards = max_categories.values.type(torch.float32)[2:]

        # dot product calculation
        net_rewards = torch.dot(torch.concat((fees,credits,max_categories_rewards),dim=-1),spending_tensor)

        # generate how to spend categories
        max_categories_card_indices = max_categories.indices
        how_to_spend = {}
        for i in range(len(max_categories_card_indices)):
            category = self.index2category[i]
            j = max_categories_card_indices[i] # j = 1,...,k
            card_i = indices[j]
            card = self.index2card[card_i]
            how_to_spend[category] = card

        
        return  {
            'value': net_rewards,
            'choices': how_to_spend
        }
    
    def test(self):
        a = self.tensor
        a = a.max(dim=0)
        print(a)
        
    

if __name__ == '__main__':
    # maps str category to constant c

    A = CardMatrix('credit_cards.csv')

    def max_comboing(card_matrix,spending,whitelist=None,blacklist=None,k=5):
        if blacklist == None:
            blacklist = set()
        if whitelist == None:
            whitelist = []

        cards = [c for c in list(A.get_card_indices().keys())  if c not in blacklist ]
        card_combinations = combinations(cards,k)
        card_combinations_filtered = [c for c in card_combinations if all([ x in c for x in whitelist ]) ]

        best_combo, best_results, max_val = None,0,0
        for combo in card_combinations_filtered:
            results = card_matrix.eval_cards(spending,combo)
            val = results['value']
            if val > max_val:
                best_combo = combo
                best_results = results
                max_val = val 
        return best_combo,best_results
    
    ###################################
    # CHANGE ME SECTION BEGINS
    ####################################

    spending =  {'Fee': -1,  # % of fees to consider
                 'Bonus Offer Value': 0,  # % bonus offer consideration
                 'Credit': 0.5, # % credit offer to be used (ie 120 uber cash for gold amex)
                 'Flights': 1200, 
                 'Hotels & Car Rentals': 300, 
                 'Other Travel': 200, 
                 'Transit': 40, 
                 'Restaurants': 400*12, 
                 'Streaming': 30, 
                 'Online Retail': 1000, 
                 'Groceries': 400*12, 
                 'Wholesale Clubs': 300, 
                 'Gas': 0, 
                 'EV Charging': 480, 
                 'Drugstores': 150, 
                 'Home Utilities': 250*12, 
                 'Cell Phone Provider': 60*12, 
                 'Rent': 24000, 
                 'All': 2000, 
                 'Choice': 0, 
    }

    # blacklist / whitelist cards
    blacklist = set(['CFU+']) # cards to exclude 
    whitelist = [] # cards that must be included 
    k = 5 # num cards

    ###################################
    # CHANGE ME SECTION ENDS
    ####################################


    combo,results = max_comboing(A,spending,whitelist=whitelist,blacklist=blacklist,k=k)
    val = results['value']
    choices = results['choices']
    total = sum(spending.values())

    print('card combo: ',combo)
    print('spent: ', total)
    print('rewards: ',val)
    print(f'% back: {val/total*100}%')
    print(f'categories to spend on: {choices}')
