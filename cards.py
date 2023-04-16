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
    spending =  {'Fee': -1,  #keep
                 'Bonus Offer Value': 0,  #keep
                 'Credit': 1, #keep
                 'Flights': 1000, 
                 'Hotels & Car Rentals': 100, 
                 'Other Travel': 600, 
                 'Transit': 30, 
                 'Restaurants': 3000, 
                 'Streaming': 40, 
                 'Online Retail': 400, 
                 'Groceries': 400*4, 
                 'Wholesale Clubs': 0, 
                 'Gas': 300*4, 
                 'EV Charging': 0, 
                 'Drugstores': 100, 
                 'Home Utilities': 200*12, 
                 'Cell Phone Provider': 60*12, 
                 'Rent': 0, 
                 'All': 300, 
                 'Choice': 500, 
    }


    A = CardMatrix('credit_cards.csv')

    def max_comboing(card_matrix,k=5):
        blacklist = set()
        cards = [c for c in A.get_card_indices().keys() if c not in blacklist ]

        card_combinations = combinations(cards,k)

        max_combo, max_val = 0,0
        for combo in card_combinations:
            val = A.eval_cards(spending,combo)
            if val > max_val:
                max_combo = combo
                max_val = val
        return max_combo,max_val
    
    print(max_comboing(A))

