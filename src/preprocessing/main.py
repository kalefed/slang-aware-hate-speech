from api.urban_dict import get_word_definition, get_slang_words, slang_words_def
import csv

word_list = get_slang_words("ok boomer")
word_def_dict = slang_words_def(word_list)
print(word_def_dict)
