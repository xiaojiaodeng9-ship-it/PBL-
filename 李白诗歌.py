# 导入核心库
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import re
import os
from wordcloud import WordCloud
import numpy as np
from PIL import Image

# ---------------------- 第一步：定义古诗情感词典（适配李白诗歌） ----------------------
# 核心情感词库（针对唐诗，尤其是李白诗歌的情感特征）
emotion_dict = {
    # 正向情感（豪放、愉悦、洒脱、壮志等）
    'positive': {
        '欢': 2, '乐': 2, '醉': 1.5, '豪': 2, '壮': 2, '畅': 2, '明': 1, '清': 1,
        '悠': 1.5, '雄': 2, '狂': 2, '扬': 1.5, '舒': 1.5, '欣': 2, '喜': 2, '逸': 2,
        '酣': 1.5, '畅': 2, '腾': 1.5, '骄': 1, '昂': 1.5, '志': 1.5, '壮': 2, '阔': 1.5,
        '闲': 1, '适': 1, '安': 1, '宁': 1, '祥': 1, '和': 1
    },
    # 负向情感（愁绪、孤寂、悲愤、迷茫等）
    'negative': {
        '愁': 2, '悲': 2, '寂': 1.5, '寞': 1.5, '难': 1.5, '茫': 2, '寒': 1, '孤': 2,
        '苦': 2, '怨': 2, '哀': 2, '凄': 2, '怅': 2, '忧': 2, '愤': 2, '叹': 1.5,
        '泣': 2, '伤': 2, '恨': 2, '愁': 2, '惘': 2, '累': 1, '空': 1, '冷': 1
    },
    # 中性情感（无明显倾向，仅记录）
    'neutral': ['山', '水', '天', '地', '风', '月', '云', '酒', '剑', '舟', '路', '城', '江', '河']
}


# ---------------------- 第二步：读取文件 ----------------------
def read_all_poem_content():
    """读取文件中所有内容，区分单首诗（按空行分隔）+ 整体内容"""
    file_names = ["李白诗歌20首.txt", "李白诗歌20首"]
    target_file = None
    for fname in file_names:
        if os.path.exists(fname):
            target_file = fname
            break
    if not target_file:
        print("❌ 未找到「李白诗歌20首」文件！")
        return "", []

    # 多编码读取
    encodings = ['utf-8', 'gbk', 'gb2312', 'ansi']
    all_text = ""
    single_poems = []  # 存储单首诗列表
    for encoding in encodings:
        try:
            with open(target_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"✅ 用 {encoding} 编码读取成功")
            break
        except:
            continue
    if not lines:
        print("❌ 文件读取失败！")
        return "", []

    # 解析单首诗（按空行分隔）+ 拼接整体内容
    current_poem = ""
    for line in lines:
        line = line.strip()
        if not line:  # 空行分隔单首诗
            if current_poem:
                single_poems.append(current_poem)
                all_text += current_poem
                current_poem = ""
            continue
        # 过滤标题/无关字符，只保留诗歌内容
        if not line.startswith(('标题：', '【', '《', '李白')):
            current_poem += line

    # 处理最后一首诗
    if current_poem:
        single_poems.append(current_poem)
        all_text += current_poem

    # 清理整体文本（只保留中文）
    all_text = re.sub(r'[^\u4e00-\u9fa5]', '', all_text)
    print(f"✅ 读取到 {len(single_poems)} 首诗，整体有效中文长度：{len(all_text)} 字")
    return all_text, single_poems


# 读取文本（整体+单首）
poem_content, single_poems = read_all_poem_content()
if not poem_content or not single_poems:
    exit()

# ---------------------- 第三步：分词+过滤 ----------------------
# 停用词表
stopwords = {
    '的', '了', '在', '是', '我', '有', '不', '人', '都', '一', '而', '也', '之', '以', '于', '者', '为', '无', '与', '尔', '其', '且', '若',
    '即',
    '君', '吾', '汝', '何', '安', '皆', '但', '只', '复', '尚', '又', '更', '还', '已', '曾', '将', '须', '应', '便', '可', '使', '令', '能',
    '兮', '哉', '乎', '焉', '欤', '耶'
}


# 整体文本分词
def text_process(text):
    """文本预处理：分词+过滤停用词"""
    words = jieba.lcut(text, cut_all=False)
    filtered_words = [word for word in words if word not in stopwords and len(word) >= 1]
    return filtered_words


# 整体词汇
total_filtered_words = text_process(poem_content)
# 单首诗词汇列表
single_poems_words = [text_process(poem) for poem in single_poems]

print(f"✅ 整体分词后有效词汇数：{len(total_filtered_words)} 个")
if len(total_filtered_words) == 0:
    print("❌ 无有效词汇可分析！")
    exit()


# ---------------------- 第四步：情感分析核心函数 ----------------------
def analyze_emotion(words_list):
    """
    情感分析函数：计算情感得分和倾向
    返回：情感倾向、正向得分、负向得分、正向词列表、负向词列表
    """
    pos_score = 0  # 正向情感总分
    neg_score = 0  # 负向情感总分
    pos_words = []  # 正向情感词
    neg_words = []  # 负向情感词

    # 统计情感词和得分
    for word in words_list:
        if word in emotion_dict['positive']:
            pos_score += emotion_dict['positive'][word]
            pos_words.append(word)
        elif word in emotion_dict['negative']:
            neg_score += emotion_dict['negative'][word]
            neg_words.append(word)

    # 判断情感倾向
    total_emotion_score = pos_score + neg_score
    if total_emotion_score == 0:
        emotion_tendency = "中性（无明显情感倾向）"
    elif pos_score > neg_score:
        pos_ratio = round(pos_score / total_emotion_score, 2)
        emotion_tendency = f"正向（豪放/愉悦），正向占比 {pos_ratio}"
    elif neg_score > pos_score:
        neg_ratio = round(neg_score / total_emotion_score, 2)
        emotion_tendency = f"负向（愁绪/孤寂），负向占比 {neg_ratio}"
    else:
        emotion_tendency = "中性（正负情感均衡）"

    return emotion_tendency, pos_score, neg_score, pos_words, neg_words


# ---------------------- 第五步：执行情感分析 ----------------------
# 1. 整体情感分析
total_emotion, total_pos_score, total_neg_score, total_pos_words, total_neg_words = analyze_emotion(
    total_filtered_words)

# 2. 单首诗情感分析
single_emotions = []
for i, poem_words in enumerate(single_poems_words):
    emotion, pos_score, neg_score, pos_words, neg_words = analyze_emotion(poem_words)
    single_emotions.append({
        'poem_index': i + 1,
        'emotion': emotion,
        'pos_score': pos_score,
        'neg_score': neg_score,
        'pos_words': pos_words,
        'neg_words': neg_words
    })

# ---------------------- 第六步：词频统计 ----------------------
word_freq = Counter(total_filtered_words)
top_n = 20
top_words = word_freq.most_common(top_n)

# 打印词频结果
print("\n" + "=" * 80)
print(f"📊 李白20首诗歌词频统计（前{top_n}）")
print("=" * 80)
for i, (word, count) in enumerate(top_words, 1):
    print(f"第{i:2d}位：{word:<4} → 出现 {count} 次")
print("=" * 80)

# ---------------------- 第七步：输出情感分析结果 ----------------------
print("\n" + "=" * 80)
print("❤️ 李白20首诗歌整体情感分析结果")
print("=" * 80)
print(f"整体情感倾向：{total_emotion}")
print(f"正向情感总分：{total_pos_score} 分（核心词：{list(set(total_pos_words))[:10]}）")
print(f"负向情感总分：{total_neg_score} 分（核心词：{list(set(total_neg_words))[:10]}）")
print(f"正向情感词数量：{len(total_pos_words)} 个")
print(f"负向情感词数量：{len(total_neg_words)} 个")
print("=" * 80)

# 单首诗情感分析（前10首，避免输出过长）
print("\n" + "=" * 80)
print("❤️ 单首诗歌情感分析（前10首）")
print("=" * 80)
for i in range(min(10, len(single_emotions))):
    emo = single_emotions[i]
    print(f"第{emo['poem_index']}首诗：{emo['emotion']}")
    print(f"  正向得分：{emo['pos_score']}（词：{emo['pos_words']}）")
    print(f"  负向得分：{emo['neg_score']}（词：{emo['neg_words']}）")
    print("-" * 50)


# ---------------------- 第八步：可视化（柱状图+词云+情感得分图） ----------------------
# 1. 词频柱状图
def plot_word_freq_bar(top_words):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    words = [item[0] for item in top_words]
    counts = [item[1] for item in top_words]

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(
        words,
        counts,
        color=plt.cm.Set2(np.linspace(0, 1, len(words))),
        edgecolor='white',
        linewidth=1,
        alpha=0.85
    )

    # 数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.2,
            f'{int(height)}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    ax.set_title('李白20首诗歌高频词汇统计', fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('核心词汇', fontsize=14, labelpad=15)
    ax.set_ylabel('出现次数', fontsize=14, labelpad=15)

    # x轴标签设置
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
        label.set_rotation_mode('anchor')

    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig('李白诗歌词频柱状图.png', dpi=300, bbox_inches='tight')
    print("✅ 词频柱状图已保存")


# 2. 情感得分对比图
def plot_emotion_score(single_emotions):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 取前10首诗做对比
    poem_indices = [emo['poem_index'] for emo in single_emotions[:10]]
    pos_scores = [emo['pos_score'] for emo in single_emotions[:10]]
    neg_scores = [emo['neg_score'] for emo in single_emotions[:10]]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(poem_indices))
    width = 0.35

    # 绘制正负向得分柱状图
    ax.bar(x - width / 2, pos_scores, width, label='正向情感得分', color='#2E8B57', alpha=0.8)
    ax.bar(x + width / 2, neg_scores, width, label='负向情感得分', color='#DC143C', alpha=0.8)

    # 设置标签
    ax.set_title('李白诗歌单首情感得分对比（前10首）', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('诗歌序号', fontsize=14)
    ax.set_ylabel('情感得分', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(poem_indices)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig('李白诗歌情感得分图.png', dpi=300, bbox_inches='tight')
    print("✅ 情感得分图已保存")


# 3. 词云图
def plot_wordcloud(filtered_words):
    word_text = ' '.join(filtered_words)
    font_path = 'simhei.ttf' if os.name == 'nt' else '/Library/Fonts/SimHei.ttf'
    if not os.path.exists(font_path) and os.name != 'nt':
        font_path = None

    wc = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color='white',
        max_words=100,
        max_font_size=200,
        min_font_size=10,
        colormap='Set2',
        random_state=42
    )
    wc.generate(word_text)

    plt.figure(figsize=(14, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('李白20首诗歌核心词汇词云图', fontsize=20, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.show()
    wc.to_file('李白诗歌词云图.png')
    print("✅ 词云图已保存")


# 执行可视化
plot_word_freq_bar(top_words)
plot_emotion_score(single_emotions)
plot_wordcloud(total_filtered_words)

# ---------------------- 第九步：补充统计信息 ----------------------
total_words = len(total_filtered_words)
unique_words = len(word_freq)
print("\n" + "=" * 80)
print("📝 补充统计信息")
print("=" * 80)
print(f"总有效词汇数：{total_words} 个")
print(f"唯一词汇数：{unique_words} 个")
print(f"出现次数最多的词汇：{top_words[0][0]}（{top_words[0][1]} 次）")
print(f"整体正向情感词占比：{round(len(total_pos_words) / (len(total_pos_words) + len(total_neg_words)) * 100, 2)}%" if (
                                                                                                                          len(total_pos_words) + len(
                                                                                                                      total_neg_words)) > 0 else "无情感词")
print("=" * 80)