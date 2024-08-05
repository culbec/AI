from collections import defaultdict, deque, Counter
import string
import random

class Markov(object):
    def __init__(self, no_states: int, is_file=True):
        self.state_size = no_states
        self.is_file = is_file
        
        self.text = None
        self.model = None
        
    def __read_text(self, file_path):
        """
            Reads a text from a file.
            
            :param file_path: The path to the readable file.
            :rtype: str
            :return: The read text from a file.
        """
        text = []
        
        with open(file_path, 'r') as file:
            for line in file:
                text.append(line.strip())
            
        return ' '.join(text)
    
    def __remove_punctuations(self, text: str):
        """
            Removes the punctuations of a given text.
            
            :param text: Text to remove the punctuations of.
            :rtype: str
            :return: The same text without any punctuations.
        """
        return text.translate(str.maketrans('','', string.punctuation))
    
    def __train(self):
        """
            Trains the model by creating a dictionary with a word as key and
            a list of next words as value.
            
            Example: hello sherlock is some is nice nice is pip
            =>  {   
                    'hello' : ['sherlock'],
                    'sherlock' : ['is'],
                    'is' : ['some', 'nice', 'pip'],
                    'some' : ['is'],
                    'nice' : ['nice', 'is']
                    'pip' : []
                }
        """
        
        words = self.text.split(' ')
        m_dict = defaultdict(lambda: (deque(maxlen=self.state_size), Counter()))

        for curr_w, next_w in zip(words[0:-1], words[1:]):
            m_dict[curr_w][0].append(next_w)
            m_dict[curr_w][1][next_w] += 1

            # Removing the least frequent state if the state size is exceeded.
            if len(m_dict[curr_w][0]) > self.state_size:
                least_frequent_word = m_dict[curr_w][1].most_common()[:-self.state_size-1:-1][0][0]
                del m_dict[curr_w][1][least_frequent_word]

        return dict(m_dict)
    
    def generate(self, _input, first_word, no_words=20):
        """
            Generates a text based on the passed first word and number of states in the text.
            
            :param _input: The input on which the Markov chain will be built.
            :param first_word: The first word of the text.
            :param no_words: The number of words in the text.
            
            :rtype: str
            :return: The generated text based on the trained model.
        """
        if self.model is None:
            if self.is_file:
                self.text = self.__remove_punctuations(self.__read_text(_input))
            else:
                self.text = self.__remove_punctuations(_input)
            
            self.model = self.__train()

        if first_word is not None and first_word not in list(self.model.keys()):
            return "Unknown word!"
        
        if first_word is None:
            word = random.choice(list(self.model.keys()))
        else:
            word = str(first_word)
            
        predicted = '' + word.capitalize()
        
        for i in range(no_words - 1):
            if word not in list(self.model.keys()):
                break

            next_word = random.choices(
                list(self.model[word][1].keys()),
                weights=list(self.model[word][1].values())
            )[0]
            word = next_word

            predicted += ' ' + next_word.lower()

            if (i + 1) % 5 == 0:
                predicted += '\n'
            
        return predicted
        
    
    