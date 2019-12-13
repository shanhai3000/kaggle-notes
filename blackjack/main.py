def count(cards):
    """
    计算手牌点数
    cards = ['A', 'K'] ---> 21点
    """
    card_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10,
                "K": 10, "A": 11}
    total_count = 0
    for card in cards:
        if card in card_map:
            total_count += card_map[card]
    for i in range(cards.count("A")):
        if total_count > 21:
            total_count -= 10

    if total_count > 21:
        print("exploded")
    return total_count


print(count(["A", "J", "5", "10"]))
