def parse(sg_obj, verbose=False):
    _object_classes = ["__background__", "bush", "kite", "pant", "laptop", "paper", "shoe", "chair", "ground", "tire",
                       "cup", "sky", "bench", "window", "bike", "board", "orange", "hat", "hill", "plate", "woman",
                       "handle", "animal", "food", "bear", "wave", "giraffe", "background", "desk", "foot", "shadow",
                       "lady", "shelf", "bag", "sand", "nose", "rock", "sidewalk", "motorcycle", "fence", "people",
                       "house",
                       "sign", "hair", "street", "zebra", "mirror", "racket", "logo", "girl", "arm", "flower", "leaf",
                       "clock", "dirt", "boat", "bird", "umbrella", "leg", "bathroom", "surfer", "water", "sink",
                       "trunk",
                       "post", "tower", "box", "boy", "cow", "skateboard", "roof", "pillow", "road", "ski", "wall",
                       "number", "pole", "table", "cloud", "sheep", "horse", "eye", "top", "neck", "tail", "vehicle",
                       "banana", "fork", "head", "door", "bus", "glass", "train", "child", "line", "ear", "reflection",
                       "car", "tree", "bed", "cat", "donut", "cake", "grass", "toilet", "player", "airplane", "ocean",
                       "glove", "helmet", "shirt", "floor", "bowl", "snow", "couch", "field", "lamp", "book", "branch",
                       "elephant", "tile", "beach", "pizza", "wheel", "picture", "plant", "sandwich", "mountain",
                       "track",
                       "hand", "plane", "stripe", "letter", "skier", "vase", "man", "building", "short", "surfboard",
                       "phone", "light", "counter", "dog", "face", "jacket", "person", "part", "truck", "bottle",
                       "jean",
                       "wing"]
    _predicate_classes = ["__background__", "and", "in_a", "cover", "over", "at", "have", "in", "carry", "rid",
                          "have_a",
                          "inside_of", "wear_a", "for", "in_front_of", "hang_on", "on_top_of", "below", "eat", "beside",
                          "behind", "above", "under", "on_front_of", "lay_on", "around", "on_a", "look_at", "sit_on",
                          "between", "watch", "wear", "walk_on", "be_in", "along", "hold", "with", "by", "stand_on",
                          "on",
                          "next_to", "on_side_of", "attach_to", "of", "inside", "be_on", "hang_from", "near", "sit_in",
                          "stand_in", "of_a"]

    obj_cls = sg_obj['objects']['class']
    # obj_score = sg_obj['objects']['scores']
    rels = sg_obj['relationships']

    parsed_obj = {'obj': [], 'rel': [], 'attr': []}
    r_set = set()

    extracted_rels = []
    for rel in rels:
        sub_id, obj_id, pred, rel_score = rel
        sub = _object_classes[obj_cls[sub_id]]
        obj = _object_classes[obj_cls[obj_id]]
        pred = _predicate_classes[pred]

        if (sub, pred, obj) in r_set:
            continue
        r_set.add((sub, pred, obj))
        if rel_score < 0.001:
            continue
        if sub == obj:
            continue
        extracted_rels.append((sub, pred, obj, rel_score))

    extracted_rels.sort(key=lambda x: x[-1], reverse=True)

    if verbose:
        for r in extracted_rels:
            print(r)
        print('total {} relations'.format(len(extracted_rels)))

    return extracted_rels