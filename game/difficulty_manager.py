def apply_difficulty(game, action):
    """
    action:
    0 -> increase difficulty
    1 -> decrease difficulty
    2 -> no change
    """

    if action == 0:
        game.clock.tick(15)
    elif action == 1:
        game.clock.tick(5)
    else:
        game.clock.tick(10)
