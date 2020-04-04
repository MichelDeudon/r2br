import os
import re
import numpy as np
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
import matplotlib.pyplot as plt


symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%', '-', '&']

non_vegan_meat_substring = ['bacon', 'beef', 'chicken', 'lamb', 'veal', 'chorizo',
                           'steak', 'duck', 'meatball', 'andouille', 'merguez', 'porc', 'pork', 'sausag',
                            'turkey', 'tripe', 'snail', 'boudin', 'foie gras', 'horse', 'carn', 'angus', 'asadero',
                            'banger', 'boar', 'boston butt', 'branzino', 'breast', 'bresaola', 'brisket', 'chevre', 'capon',
                            'chitterlings', 'cutlet', 'deli ham', 'egg white', 'eggnog', 'escargot', 'fillet', 'goose', 'gravy',
                            'ham hock', 'hamburger', 'kidney', 'kielbasa', 'lard', 'liver', 'loin', 'lumpia skin', 'lumpia wrapper',
                            'meat', 'medium egg', 'mutton', 'pancetta', 'pheasant', 'prosciutto', 'quail', 'rabbit', 'ragu', 
                            'sheep', 'suckling pig', 'won ton wrapper', 'wonton wrapper']

non_vegan_fish_substring = ['anchov','mussel', 'scallop', 'tuna', 'salmon', 'fish',  'crab', 'caviar', 'prawn', 'shrimp', 'honey',  
                             'trout', 'yellowfin', 'yellowtail', 'wish bone', 'turbot', 'turtle', 'tentacle', 'squid', 'shark', 
                            'mackerel', 'lobster',  'bonito', 'boquerones', 'octopus', 'tongue', 'albacore', 'lingcod', 
                            'carp', 'gravlax', 'haddock', 'halibut', 'kipper', 'merluza', 'mullet', 'oyster', 'oxtail',
                            'sardine', 'sea bass', 'saba', 'tilapia']

non_vegan_dairy_substring = ['asiago', 'burrata', 'beurre', 'camembert', 'caciotta', 'cheese', 'mozarella', 'parmesan',
                             'queso', 'raclette', 'gorgonzola', 'gouda', 'grana padano', 'grated pecorino', 'parmagiano', 'parmigiana',
                             'parmigiano', 'roquefort', 'philadelphia  creme', 'cajeta', 'cantal', 'feta', 'fontina', 'fromage', 'khoa']

non_vegan_other_substring = ['honey']

non_vegan_match = ['bass', 'ham', 'egg', 'cod', 'pig', 'hen', 'dog', 'rib', 'sole'] # <= 4 caracters

adjectives = ['american', 'active', 'sauce', 'seasoning', 'chopped', 'cooked', 'dry', 'short', 'asian',
      'dried', 'fried', 'extravirgin', 'large', 'frozen', 'fresh', 'lowfat', 'glutenfree', 'neutral', 'kosher',
      'fat', 'free', 'whole', 'unsweetened', 'nosaltadded', 'half', 'cooking', 'creole', 'greek',
      'reducedfat', 'reduced', 'sodium', 'powder', 'seed', 'allpurpose', 'all purpose', 'baby', 'chinese', 'frying',
      'italian', 'diced', 'purple', 'verde', 'warm', 'hot', 'accompaniment', 'ajinomoto', 'extra']


def is_vegan(foodlist):
    '''
    Check if a basket of food (list of ingredients) is vegan.

    Args:
        foodlist: list of str, ex: ['leek', 'beef']
    Returns:
        is_vegan: bool
    '''

    for food in foodlist:
        forbidden_substrings = non_vegan_meat_substring + non_vegan_fish_substring + non_vegan_dairy_substring + non_vegan_other_substring
        for food_ in forbidden_substrings:
            if food_ in food.lower():
                return False
        for food_ in non_vegan_match: 
            if food_ == food.lower():
                return False
    return True


def clean(ingredient):
    ''' 
    Clean input string.

    Args:
        ingredient: str, ex: 'Tomatoes'
    Returns: 
        ingredient: str, ex: 'tomato'
    '''

    ingredient = ingredient.lower() # lower case
    ingredient = re.sub(r'\([^)]*\)', '', ingredient) # remove parenthesis
    for symbol in symbols:
        ingredient = ingredient.replace(symbol,'') # remove symbol
    ingredient = ' '.join([lemmatizer.lemmatize(word) for word in ingredient.split()]) # lemmatize words
    for word in adjectives:
        ingredient = ingredient.replace(word,'') # remove adjectives
    ingredient = ingredient.strip() # strip
    return ingredient


def preprocess(ingredients_list, char_min_len=3):
    '''
    Preprocess a basket of food (list of ingredients).

    Args:
        ingredients_list: list, ex: ['Tomatoes', 'Basil']
        char_min_len: int
    Returns:
        output_list: list, ex: ['tomato', 'basil']
    '''

    output_list = list(map(clean, ingredients_list))
    output_list = [food for food in output_list if len(food)>=char_min_len] 
    return output_list


def decompose_vocab(vegan_vocab, fixed_points):
    '''
    Remap vocab (parent-child relationship) using substring matching.
    
    Args:
        vegan_vocab: iterable (list, set). Vegan vocabulary (preprocessed, is_vegan)
        fixed_points: iterable (list, set). Known fruits or vegetables names, ex: 'asparagus' and 'tomato'.
    Returns:
        roots: dict, e.g. roots['cs olive oil'] = 'olive'
    '''

    parents = {} # build forest (node, parent=None)
    for w1 in vegan_vocab:
        for w2 in vegan_vocab:
            if w1 not in fixed_points and w2!=w1 and w2 in w1:
                if w1 not in parents:
                    parents[w1] = w2 # w1 does not have a parent yet and is set to w2 (shorter string)
                elif w2 in fixed_points or len(parents[w1].split())>len(w2.split()): 
                    parents[w1] = w2 # replace parent of w1 by w2 if w2 is fixed point or w2 has less words

    roots = {} # use root node as representation of a tree
    for k,v in parents.items():
        while v in parents:
            v = parents[v]
        roots[k] = v
    return roots


def plot_histograms(x1, x2, title1='', title2='', bins1=10, bins2=10, filename='example_histograms.png'):
    """ 
    Plot and save two histograms next to each other.

    Args:
        x1, x2: list
        title1, title2: string
        bins1, bin2: int
        filename: str
    """

    fig = plt.figure(1,figsize=(15,5))
    plt.subplot(121)
    plt.hist(x1, bins=bins1, color='black')
    plt.xlim(0,30)
    plt.xlabel('N ingredients')
    plt.ylabel('Counts')
    plt.title(title1)
    plt.subplots_adjust(wspace=None, hspace=None)

    plt.subplot(122)
    plt.hist(x2, bins=bins2, color='black')
    plt.xlim(0,30)
    plt.xlabel('N ingredients')
    plt.ylabel('Counts')
    plt.title(title2)
    plt.subplots_adjust(wspace=None, hspace=None)
    fig.tight_layout()
    plt.savefig(os.path.join('img/', filename))
    plt.show()


def plot_co2_estimates(threshold=10., filename='food_contrib_co2.png'):
    """ 
    Plot and save CO2 estimates.

    Args:
        threshold: int
        filename: str
    """

    # from http://www.bonpourleclimat.org/wp-content/uploads/2015/05/Charte-Restaurant-Bon-pour-le-climat-2017.pdf
    co2_emissions = [0.3, 0.7, 20*0.3, 20*0.7, 4.5, 9.8, 22.1, 33.3]
    co2_indices = [0, 0, 1, 1, 2, 3, 4, 4]
    food_type = ['Local-seasonal', '', 'Imported,\n out-of-season', '',
                 'Local chicken', 'Local butter', 'Beef', '']

    fig = plt.figure(1,figsize=(10,5))
    plt.scatter(co2_indices, co2_emissions, c='g', s=50) # kg CO2 / kg produit brut
    plt.plot([threshold]*len(co2_emissions), '--', color='black', label='10 kg CO2/kg') # threshold
    plt.axvline(x=1.5, linestyle='-', color='grey')
    plt.annotate(' Fruits \n & Vegetables', (0.05,25.), size=20)
    plt.annotate('Non vegan', (2.0,25.), size=18)

    plt.xticks(co2_indices,food_type, size=14, rotation=40)
    plt.ylabel('kg CO2 / kg brut product', size=16)
    plt.xlim(-0.25,5.25)
    plt.legend()
    fig.tight_layout()
    plt.title('Food contributions to CO2 emissions and vegan subset for this study \n * local = produced within 250km')
    plt.savefig(os.path.join('img/', filename))
    plt.show()
    

def plot_food_bias_seasons_Paris(fruits_and_vegs, fruits_and_vegs_bias, ranked_idx, seasons, month=11, n_subplots=4, 
                                 filename='food_bias_season{}_Paris.png', figsize=(15,5)):
    """ 
    Plot and save fruit/vegs bias and seasonality in Paris.

    Args:
        fruits_and_vegs, fruits_and_vegs_bias: list or set
        ranked_idx: np.array
        seasons: dict, e.g. seasons[11] is an iterable for Paris, December
        month: int
        n_subplots: int
        filename: str
        figsize: tuple
    """

    n_fruits_vegs = len(ranked_idx)
    n_rows = n_fruits_vegs//n_subplots
    tail = n_fruits_vegs - n_rows*n_subplots

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    fig = plt.figure(1,figsize=figsize)
    for i in range(n_subplots):
        plt.subplot(1,n_subplots,1+i)
        food_names = fruits_and_vegs[ranked_idx][tail+i*n_rows:tail+(i+1)*n_rows]
        food_bias = fruits_and_vegs_bias[ranked_idx][tail+i*n_rows:tail+(i+1)*n_rows]
        markers = ['o' if food in seasons[month] else 'x' for food in food_names] # 'o' if local-seasonal in december, else '+'    
        colors = ['green' if m=='o' else 'gray' for m in markers]
        alphas = [1.0 if m=='o' else 0.5 for m in markers]
        
        for k, (food_name, bias, m, c, alpha) in enumerate(zip(food_names, food_bias, markers, colors, alphas)):
            x, y = bias, k
            plt.scatter(x,y, marker=m, c=c)
            plt.annotate('  {}  ({:.2f}%)'.format(food_name.capitalize(), 100*bias), (x,y), alpha=alpha, size=16)

        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.box(on=None)
        #plt.title('Food ranked by bias, \n colored by seasonality (Paris, December)', size=14)

    fig.tight_layout()
    plt.savefig(os.path.join('img/', filename.format(month+1)))
    plt.show()

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    
    
def plot_Zipf(vocab, histogram, fruits_and_vegs_ranked, fruits_and_vegs_bias, ranked_idx, filename='Zipflaw_power.png'):
    """ 
    Plot and save empirical Zipf power law on Vegan10K corpus and fruits/vegs, as a subset.

    Args:
        vocab: set
        histogram: list
        fruits_and_vegs, fruits_and_vegs_bias: list or set
        ranked_idx: np.array
        filename: str
    """

    fig = plt.figure(1,figsize=(7,5))
    plt.scatter(np.arange(len(vocab)), np.log(histogram), c='orange', alpha=0.05, label='All ingredients')
    plt.scatter(fruits_and_vegs_ranked[ranked_idx], np.log(fruits_and_vegs_bias[ranked_idx]), alpha=1., c='green', 
                label='Fruit and vegetables')
    plt.xlabel('food rank')
    plt.xlim(0,len(vocab))
    plt.ylabel('log (word frequency)', size=11)
    plt.legend()
    plt.title('Zipf law on Vegan10K')
    fig.tight_layout()
    plt.savefig(os.path.join('img/', filename))
    plt.show()
