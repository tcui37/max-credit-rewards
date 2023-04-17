# max-credit-rewards

Purpose: 
- Provide a approximation on the optimal set of credit cards based on your spending

Assumptions made
- custom choice 'other' will automatically apply to the 'other' category. It does NOT choose a category to maximize your rewards yet (4/16/23)
- USC is automatically set to give you benefits on phone bills and home utilities due to nicheness

Usage
- blacklist: cards each combination must exclude
- whitelist: cards each combination must include
- k: number of cards

TODO:
- figure out how CSP actually works. Many redemptions are travel based, so will likely implement a new card CSP*
- Actually spread the else category to anything that may have been 0
- proper optimization on choice based cards.
