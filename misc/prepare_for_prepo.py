import json

def process_captions(sentence_list):
    sentences = []
    for sentence in sentence_list:
        sentences.append(sentence['raw'])
    return sentences

json_file = json.load(open('dataset.json'))
out_file = []
for idx, img in enumerate(json_file['images']):
    temp = {}
    temp['file_path'] =  img['filename'] #'D:/LC/Datasets/Flickr8k/Images/'
    temp['split'] = img['split']
    temp['captions'] = process_captions(img['sentences'])
    temp['id'] = idx
    out_file.append(temp)

with open('D:/LC/Datasets/data/flickr8k/flickr8k_set.json','w') as f:
    json.dump(out_file, f)
f.close()

debug100 = []
count_train = count_val = count_test = 0
for idx, img in enumerate(out_file):
    if img['split'] == 'train' and count_train < 80:
        count_train += 1
        debug100.append(img)
    if img['split'] == 'val' and count_val < 10:
        count_val += 1
        debug100.append(img)
    if img['split'] == 'test' and count_test < 10:
        count_test += 1
        debug100.append(img)
with open('D:/LC/Datasets/data/flickr8k/debug100/flickr8k_debug100_set.json', 'w') as f:
    json.dump(debug100, f)
f.close()
# coco = json.load(open('D:/LC/Datasets/coco/captions_val2014.json','r'))

# # create annotation file in coco format
# val = {}
# val['annotations'] = []
# val['images'] = []
# test = {}
# test['annotations'] = []
# test['images'] = []
# for idx, img in enumerate(out_file):
#     if img['split'] == 'val':
#         # first construct annotations, len(annotations) = len(images) * 5
#         for caption in img['captions']:
#             temp = {}
#             temp['caption'] = caption
#             temp['id'] = idx
#             temp['image_id'] = idx
#             val['annotations'].append(temp)
#         # second construct images
#         temp = {}
#         temp['file_name'] = img['file_path']
#         temp['id'] = idx
#         val['images'].append(temp)
#
#     elif img['split'] == 'test': # the same for the test split
#         for caption in img['captions']:
#             temp = {}
#             temp['caption'] = caption
#             temp['id'] = idx
#             temp['image_id'] = idx
#             test['annotations'].append(temp)
#         # second construct images
#         temp = {}
#         temp['file_name'] = img['file_path']
#         temp['id'] = idx
#         test['images'].append(temp)
#
# with open('D:/LC/Datasets/Flickr8k/flickr8k_val.json', 'w') as f:
#     json.dump(val, f)
# f.close()
#
# with open('D:/LC/Datasets/Flickr8k/flickr8k_test.json','w') as f:
#     json.dump(test, f)
# f.close()

Z = 1