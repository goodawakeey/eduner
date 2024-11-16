import requests
import time

# Stanford CoreNLP服务器地址
corenlp_url = 'http://localhost:9000'

# 函数：调用Stanford CoreNLP服务器获取词性和依存信息
def get_corenlp_info(sentence, max_retries=3):
    text = " ".join([word for word, ner in sentence])

    params = {
        'annotators': 'tokenize,ssplit,pos,ner,depparse',
        'pipelineLanguage': 'zh',
        'outputFormat': 'json'
    }

    if len(text) > 100000:
        print(f"Warning: Sentence too long to process, skipping. Length: {len(text)}")
        return None

    retries = 0
    while retries < max_retries:
        try:
            print(f"Sending request to CoreNLP for sentence: {text}")
            response = requests.post(corenlp_url, params={'properties': str(params)}, data=text.encode('utf-8'))

            if response.status_code == 200:
                print("Server response received successfully.")
                return response.json()
            else:
                print(f"Error: Received response with status code {response.status_code}")
                print(f"Response content: {response.text}")

        except requests.exceptions.Timeout:
            print("Error: Request timed out.")
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to CoreNLP server. Details: {e}")

        retries += 1
        print(f"Retrying... ({retries}/{max_retries})")
        time.sleep(2)

    print("Failed to get valid response after multiple retries.")
    return None

# 函数：简化依存标签
def simplify_dependency_label(label):
    if 'compound' in label:
        return 'nn'
    if label == 'ROOT':
        return 'root'
    if ':' in label:
        return label.split(':')[0]
    return label

# 函数：将 CoNLL 转换为 CoNLL-X
def process_conll_to_conllx(input_file, output_file):
    sentences = []
    sentence = []

    # 读取 CoNLL 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if not line:  # 空行表示句子结束
                if sentence:
                    sentences.append(sentence)
                sentence = []
            else:
                word, ner = line.split()
                sentence.append((word, ner))

    if sentence:
        sentences.append(sentence)

    # 处理每个句子并写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            if not sentence:
                continue

            # 获取CoreNLP的词性和依存信息
            corenlp_result = get_corenlp_info(sentence)

            if corenlp_result is None:
                print("Skipping sentence due to CoreNLP processing error.")
                continue

            # 获取依存关系的映射
            dependencies = {dep['dependent']: dep for dep in corenlp_result['sentences'][0]['basicDependencies']}

            # 处理每个词
            for token in corenlp_result['sentences'][0]['tokens']:
                word = token['word']
                pos = token['pos']

                # 获取依存关系信息
                dep_info = dependencies.get(token['index'], {})
                head = dep_info.get('governor', 0)
                dep = dep_info.get('dep', '_')

                # 简化依存标签
                dep = simplify_dependency_label(dep)

                ner = sentence[token['index'] - 1][1]

                # 写入CoNLL-X格式
                f.write(f"{token['index']}\t{word}\t-\t{pos}\t{pos}\t-\t{head}\t{dep}\t_\t_\t{ner}\n")
            f.write("\n")  # 句子结束后换行

# 调用函数处理 train、dev 和 test 数据集
if __name__ == "__main__":
    process_conll_to_conllx('dataset/train.conll', 'dataset/train.sd.conllx')
    process_conll_to_conllx('dataset/dev.conll', 'dataset/dev.sd.conllx')
    process_conll_to_conllx('dataset/test.conll', 'dataset/test.sd.conllx')