import json
from  symspellpy import * 
import zipfile
import os



def lv(str_1: str, str_2: str) -> int:
    n,m = len(str_1),len(str_2)
    if n > m:
        str_1, str_2 = str_2, str_1
        n,m = m,n

    cur = range(n+1)
    for i in range(1, m+1):
        per,cur=cur,[i]+[0]*n
        for j in range(1,n+1):
            add,delete,change = per[j]+1, cur[j-1] +1,per[j-1]
            if str_1[j-1] != str_2[i-1]:
                change += 1
            cur[j] = min(add, delete, change)
    return cur[n]


def find_simple_word(a: str) -> str:
    dif = 10*20
    f = open("../../Desktop/russian.txt").readlines()
    for i in f:
        c = i.replace('\n','')
        if (len(c)-len(a)) <= 2:
            if lv(c,a) < dif:
                dif = lv(c,a)
                ans = c
    if ans == "":
        return a
    else:
        return ans
    

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
corpus_path = r"C:\Users\Tema\Desktop\russian.txt"

print(f"Попытка загрузить словарь из: {corpus_path}")
try:
    with open(corpus_path, "r", encoding="cp1251") as f:
        for line in f:
            word = line.strip().lower() 
            if word: 
                sym_spell.create_dictionary_entry(word, 1)
    print(f"Словарь успешно загружен из: {corpus_path}. Количество слов: {sym_spell.word_count}")
except:
    pass


def correct(text_to_correct):
    suggestions = sym_spell.lookup_compound(text_to_correct, max_edit_distance=2)
    corrected_text = suggestions[0].term if suggestions else text_to_correct
    return corrected_text



path = r"C:\Users\Tema\Desktop\school"
contents = os.listdir(path)

for item in contents:
    full_path = os.path.join(path, item) 
    with open(full_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data = data['data']['text']
    print(correct(data))
    print(data)
    input()
