import json
def process_captions(sentence_list):
    sentences = []
    for sentence in sentence_list:
        sentences.append(sentence['raw'])
    return sentences

json_file = json.load(open('dataset.json'))
temp_json = json.load(open('D:/LC/Datasets/data/flickr8k/debug100/data.json'))
out_file = []
for idx, img in enumerate(json_file['images']):
    temp = {}
    temp['file_path'] = 'D:/LC/Datasets/Flickr8k/Images/' + img['filename']
    temp['split'] = img['split']
    temp['captions'] = process_captions(img['sentences'])
    temp['id'] = idx
    out_file.append(temp)

with open('D:/LC/Datasets/data/flickr8k/debug100/flickr8k_100_set_new.json','w') as f:
    json.dump(out_file, f)

temp = json.load(open('D:/LC/Datasets/data/flickr8k/debug100/flickr8k_100_set_new.json', 'r'))

Z = 1