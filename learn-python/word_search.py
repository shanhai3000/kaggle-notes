import numpy as np

planets = ['Mercury',
           'Venus',
           'Earth',
           'Malacandra',
           'Jupiter',
           'Saturn',
           'Uranus',
           'Neptune']

meals = ['Spam', 'Eggs', 'Spam', 'Bacon', 'Spam']

i = 0
n = 0
boring = None
while n < len(meals):
    if n + 1 < len(meals) and meals[n] == meals[n + 1]:
        boring = (True)
        break
    n += 1

key_word = "casino"
doc_list = ["The Learn Python Challenge Casino in Casino casino.", "They bought a car", "Casinoville Casino"]

doc_list = [phrase.split() for phrase in doc_list]
ii = []


def word_search(doc_list, key_word):
    res = []
    for idx, words in enumerate(doc_list):
        for word in words:
            word = word.lower().strip(".")
            if word == key_word and not res.count(idx):
                res.append(idx)
    return res


keywords = ['casino', 'they']
rr = {key_word: word_search(doc_list, key_word) for key_word in keywords}

a1 = [1, 2, 3, 4, 5]
a1 = np.array(a1)
print(a1 + 19)
