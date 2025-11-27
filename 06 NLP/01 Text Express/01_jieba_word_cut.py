import jieba

text = "小明毕业于北京大学计算机系"

# 精确模式
print("精确模式")
word_generator = jieba.cut(text)
for word in word_generator:
    print(word)
print(word_generator)

word_list = jieba.lcut(text)
print(word_list)
print('\n')

# 全模式
print("全模式")
word_generator = jieba.cut(text, cut_all=True)
for word in word_generator:
    print(word)

word_list = jieba.lcut(text, cut_all=True)
print(word_list)
print('\n')

# 搜索引擎模式
print("搜索引擎模式")
word_generator = jieba.cut_for_search(text)
for word in word_generator:
    print(word)

word_list = jieba.lcut_for_search(text)
print(word_list)