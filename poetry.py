# encoding=utf-8
import jieba
import jieba.posseg as pseg




# jieba.del_word('明月光')
# jieba.del_word('望明月')
# jieba.suggest_freq(('月', '光'), True)
seg_list = jieba.cut("床前明月光,疑是地上霜。举头望明月,低头思故乡。")
words =pseg.cut("床前明月光,疑是地上霜。举头望明月,低头思故乡。")
print(" ".join(seg_list))  # 全模式

# for word, flag in words:
#     print('%s %s' % (word, flag))

