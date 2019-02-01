# Hangman game
#

import random

WORDLIST_FILENAME = "words.txt"

def loadWords():
    """
    Returns a list of valid words. Words are strings of lowercase letters.
    
    Depending on the size of the word list, this function may
    take a while to finish.
    """
    print("Loading word list from file...")
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r')
    # line: string
    line = inFile.readline()
    # wordlist: list of strings
    wordlist = line.split()
    print("  ", len(wordlist), "words loaded.")
    return wordlist

def chooseWord(wordlist):
    """
    wordlist (list): list of words (strings)

    Returns a word from wordlist at random
    """
    return random.choice(wordlist)

# -----------------------------------

# Load the list of words into the variable wordlist
# so that it can be accessed from anywhere in the program
wordlist = loadWords()

def isWordGuessed(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: boolean, True if all the letters of secretWord are in lettersGuessed;
      False otherwise
    '''
    t_f = []
    for i in secretWord:
        if i in lettersGuessed:
            t_f.append(True)
        else:
            t_f.append(False)
    if False in t_f:
        return False
    else:
        return True


def getGuessedWord(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters and underscores that represents
      what letters in secretWord have been guessed so far.
    '''
    # FILL IN YOUR CODE HERE...
    word = list('_' * len(secretWord))
    for i in range(0, len(secretWord)):
        if secretWord[i] in lettersGuessed:
            word[i] = secretWord[i]
    return ''.join(word)


def getAvailableLetters(lettersGuessed):
    '''
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters that represents what letters have not
      yet been guessed.
    '''
    import string
    all_letter = list(string.ascii_lowercase)
    for i in lettersGuessed:
        all_letter.remove(i)
    return ''.join(all_letter)

def hangman(secretWord):
    '''
    secretWord: string, the secret word to guess.

    Starts up an interactive game of Hangman.

    * At the start of the game, let the user know how many 
      letters the secretWord contains.

    * Ask the user to supply one guess (i.e. letter) per round.

    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computers word.

    * After each round, you should also display to the user the 
      partially guessed word so far, as well as letters that the 
      user has not yet guessed.

    Follows the other limitations detailed in the problem write-up.
    '''
def hangman(secretWord):
    print('Welcome to the game, Hangman!')
    print('I am thinking of a word that is', len(secretWord), 'letters long.')
    print('--------------')
    NumGuess = 0
    lettersGuessed = []
    while NumGuess < 8 and '_' in getGuessedWord(secretWord, lettersGuessed):
        print('You have', 8 - NumGuess, 'guesses left.')
        print('Available letters:', getAvailableLetters(lettersGuessed))
        guess = input('Please guess a letter: ')
        guessInLowerCase = guess.lower()
        
        if guessInLowerCase in lettersGuessed:
            print("Oops! You've already guessed that letter:", getGuessedWord(secretWord, lettersGuessed))
            print('--------------')
        else:    
            if guessInLowerCase in secretWord:
                lettersGuessed.append(guessInLowerCase)
                NumGuess += 1
                print('Godd guess:', getGuessedWord(secretWord, lettersGuessed))
                print('--------------')
            else:
                NumGuess += 1
                lettersGuessed.append(guessInLowerCase)
                print('Oops! That letter is not in my word:',getGuessedWord(secretWord, lettersGuessed))
                print('--------------')
    if '_' in getGuessedWord(secretWord, lettersGuessed):
        print('Sorry, you ran out of guesses. The word was', secretWord, '.')
    else:
        print('Congratulations, you won!')



secretWord = chooseWord(wordlist).lower()
hangman(secretWord)
