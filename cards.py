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
        self.card_index = {self.data[i][0]:i-2 for i in range(2,m)}
        self.category_index = {self.data[0][i]:i-2 for i in range(2,n-1)}


    def __repr__(self) -> str:
        rep = ''
        for row in self.data:
            rep += '\n' + str(row)
        return rep
    
    def get_tensor(self): return self.tensor
    
    def get_card_indices(self): return self.card_index

    def get_category_indices(self): return self.category_index

    def eval_cards(self,spendings:dict,ids: list) -> float:
        m = len(self.category_index)
        spending_tensor = torch.zeros(m,dtype=torch.float32)
        for category,spent in spendings.items():
            spending_tensor[self.category_index[category]] = float(spent)

        indices = [self.card_index[card_id] for card_id in ids ]
        max_categories = self.tensor[indices].max(dim=0,keepdims=False).values.type(torch.float32)
        
        return torch.dot(max_categories,spending_tensor)
    

if __name__ == '__main__':
    # maps str category to constant c

    # CHANGE ME
    spending =  {'Fee': -1,  # % of fees to consider
                 'Bonus Offer Value': 0,  # % bonus offer consideration
                 'Credit': 0.5, # % credit offer to be used (ie 120 uber cash for gold amex)
                 'Flights': 1000, 
                 'Hotels & Car Rentals': 500, 
                 'Other Travel': 200, 
                 'Transit': 40, 
                 'Restaurants': 500*12, 
                 'Streaming': 10, 
                 'Online Retail': 1000, 
                 'Groceries': 3000, 
                 'Wholesale Clubs': 300, 
                 'Gas': 0, 
                 'EV Charging': 480, 
                 'Drugstores': 150, 
                 'Home Utilities': 250*12, 
                 'Cell Phone Provider': 60*12, 
                 'Rent': 24000, 
                 'All': 20000, 
                 'Choice': 0, 
    }

    # jacky_spending =  {'Fee': -1,  # % of fees to consider
    #              'Bonus Offer Value': 0,  # % bonus offer consideration
    #              'Credit': 1, # % credit offer to be used (ie 120 uber cash for gold amex)
    #              'Flights': 500, 
    #              'Hotels & Car Rentals': 0, 
    #              'Other Travel': 100, 
    #              'Transit': 40, 
    #              'Restaurants': 1200, 
    #              'Streaming': 360, 
    #              'Online Retail': 250, 
    #              'Groceries': 2400, 
    #              'Wholesale Clubs': 300, 
    #              'Gas': 0, 
    #              'EV Charging': 480, 
    #              'Drugstores': 100, 
    #              'Home Utilities': 0, 
    #              'Cell Phone Provider': 720, 
    #              'Rent': 24000, 
    #              'All': 400, 
    #              'Choice': 0, 
    # }


    


    A = CardMatrix('credit_cards.csv')

    def max_comboing(card_matrix,spending,blacklist=None,k=5):
        if blacklist == None:
            blacklist = set()
        cards = [c for c in A.get_card_indices().keys() if c not in blacklist ]

        card_combinations = combinations(cards,k)

        max_combo, max_val = 0,0
        for combo in card_combinations:
            val = card_matrix.eval_cards(spending,combo)
            if val > max_val:
                max_combo = combo
                max_val = val
        return max_combo,max_val
    
    blacklist = set()
    combo,val = max_comboing(A,spending,blacklist=blacklist,k=5)
    print(combo,val)
