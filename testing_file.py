# testing_file.py

import random

def get_random_joke():
    jokes = [
        "Why don't programmers like nature? It has too many bugs.",
        "Why did the programmer quit his job? Because he didn't get arrays.",
        "What do you call a bear with no teeth? A gummy bear!",
        "Why don't eggs tell jokes? They'd crack up!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call a dinosaur that crashes into things? Tyranno-wrecks!",
        "Why can't a nose be 12 inches long? Because then it would be a foot!",
        "What do you call a can opener that doesn't work? A can't opener!",
        "Why did the math book look so sad? Because it had too many problems."
    ]
    return random.choice(jokes)

if __name__ == "__main__":
    print("Here's a random joke for you:")
    print(get_random_joke())