import matplotlib as matplot

sample = [
    {'name': 'Peach', 'items': ['green shell', 'banana', 'green shell', ], 'finish': 3},
    {'name': 'Bowser', 'items': ['green shell', ], 'finish': 1},
    {'name': None, 'items': ['mushroom', ], 'finish': 2},
    {'name': 'Toad', 'items': ['green shell', 'mushroom'], 'finish': 1},
]


def best_items(racers):
    winner_items_count = {}
    # for i, v in enumerate(racers):
    for i in range(len(racers)):

        if racers[i]['finish'] == 1:
            for item in racers[i]['items']:
                if item not in winner_items_count:
                    winner_items_count[item] = 0
                winner_items_count[item] += 1
        if racers[i]['name'] is None:
            #     print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(
            #     i+1, len(racers), racer['name'])
            #      )
            # print("WARNING: Encountered racer with unknown name on teration {}/{}(racer = {})".format(i+1, len(racers), v['name']))
            print("WARNING: Encountered racer with unknown name on iteration {}/{} (racer = {})".format(i + 1,
                                                                                                        len(racers),
                                                                                                        racers[i][
                                                                                                            'name']))
    return winner_items_count


aa = [{'name': 'Peach', 'items': ['green shell', 'banana', 'green shell'], 'finish': 3},
      {'name': 'Peach', 'items': ['green shell', 'banana', 'green shell'], 'finish': 1},
      {'name': 'Bowser', 'items': ['green shell'], 'finish': 1}, {'name': None, 'items': ['green shell'], 'finish': 2},
      {'name': 'Bowser', 'items': ['green shell'], 'finish': 1}, {'name': None, 'items': ['red shell'], 'finish': 1},
      {'name': 'Yoshi', 'items': ['banana', 'blue shell', 'banana'], 'finish': 7},
      {'name': 'DK', 'items': ['blue shell', 'star'], 'finish': 1}]

print(best_items(aa))
