from vizwiz_api.vizwiz import VizWiz
import pandas as pd
import json

def df_from_vizwiz(MAX_ANNS, PATH_TO_ANN='data/annotations/', PATH_TO_PICTURES='data/', split_types=['train'], verbose=1):
    '''Function, which takes in the number of annotations existing for this split of pictures and the path to the json which contains the images-urls and their captions.'''
    list_of_dicts = []
    i = 0
    for split_type in split_types:
        vizwiz = VizWiz(PATH_TO_ANN + split_type+'.json', ignore_rejected=True, ignore_precanned=True) 
        while i < MAX_ANNS:
            try:
                anns = vizwiz.loadAnns(i)
                image_id = anns[0]['image_id']
                file_name = vizwiz.loadImgs(image_id)[0]['file_name']
                url = PATH_TO_PICTURES + split_type + '/' + file_name
                anns[0]['url'] = url
                anns[0]['split_type'] = split_type
                list_of_dicts.append(anns[0])
                i += 1
            except KeyError:
                if verbose > 0:
                    print('Error with Ann-ID {}'.format(i))
                i += 1
    return pd.DataFrame(list_of_dicts)


def df_from_val_json(json_url, path_to_data='data/val/'):
    '''Function, which takes in the path to the json containing the image-urls and the captions of the validation-dataset. Has a slightly different format than the train-json, thus a different function is needed.'''
    f = open(json_url)
    data = json.load(f)
    val_set = pd.merge(pd.DataFrame.from_dict(data['images']), pd.DataFrame.from_dict(data['annotations']), left_on = 'id', right_on = 'image_id').copy()
    val_set['url'] = path_to_data + val_set['file_name']
    return val_set[['url', 'caption']].copy()