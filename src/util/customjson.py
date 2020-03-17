import json
import numpy as np
from functools import lru_cache

SERIALIZEABLE_FLAG = '_t'

class JSONSerializable:
    """
    custom class that can be easily serialized to JSON and deserialized from JSON
    sub class should be able to be constructed with no parameters

    after the class is defined, register_cls should be called
    """
    cls_info = {}
    cls_name_map = {}

    def __init__(self):
        pass

    def serialize(self):
        class_info = JSONSerializable.cls_info.get(type(self))
        if class_info:
            data = {SERIALIZEABLE_FLAG: class_info['class_name']}
            attr_map = class_info['attr_map']
            for k, v in self.__dict__.items():
                data[attr_map.get(k, k)] = v
            return data
        else:
            data = {SERIALIZEABLE_FLAG: self.__class__.__name__}
            for k, v in self.__dict__.items():
                data[k] = v
            return data

    @staticmethod
    def deserialize(d):
        class_name = d[SERIALIZEABLE_FLAG]

        cls = JSONSerializable.cls_name_map[class_name]
        cls_info = JSONSerializable.cls_info[cls]

        name_map = cls_info['name_map']
        obj = cls()
        for k, v in d.items():
            if k == SERIALIZEABLE_FLAG:
                pass
            obj.__dict__[name_map.get(k, k)] = v
        return obj

    @staticmethod
    def register_cls(cls, class_name=None, attr_abbreviation={}):
        """

        :param cls: another class that inherited this class
        :param class_name: abbreviation for the name of the class, used in serialize/deserialize
        :param attr_abbreviation: abbreviation for the name of class members, used in serialize/deserialize
        :return: None
        """
        if class_name is None:
            class_name = cls.__name__

        assert cls not in JSONSerializable.cls_info, \
            'class {} has been registered'.format(cls.__name__)
        assert class_name not in JSONSerializable.cls_name_map, \
            'name {} has been registered for class {}'.format(class_name,
                                                              JSONSerializable.cls_name_map[class_name].__name__)
        # kept field
        assert SERIALIZEABLE_FLAG not in attr_abbreviation.keys() and SERIALIZEABLE_FLAG not in attr_abbreviation.values(), \
            'illegal field name or abbreviation used'

        info = {'class_name': class_name, 'attr_map': attr_abbreviation}
        name_map = dict((v, k) for k, v in attr_abbreviation.items())
        assert len(attr_abbreviation) == len(name_map), 'attr_map invalid'
        info['name_map'] = name_map

        assert cls not in JSONSerializable.cls_info
        assert cls not in JSONSerializable.cls_name_map
        JSONSerializable.cls_info[cls] = info
        JSONSerializable.cls_name_map[class_name] = cls

def _json_serialize(obj):
    """
    supports serialize of JSONSerializable objects and numpy arrays
    :param obj:
    :return:
    """
    if isinstance(obj, JSONSerializable) or \
            obj.__class__.__name__ in JSONSerializable.cls_info:
        return obj.serialize()
    elif isinstance(obj, np.int_):
        return int(obj)
    elif isinstance(obj, np.float_):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def _json_deserialize(obj):
    if "_t" in obj:
        return JSONSerializable.deserialize(obj)
    else:
        return obj

def load_custom(fp, **kwargs):
    if isinstance(fp, str):
        with open(fp, 'r') as f:
            return json.load(f, object_hook=_json_deserialize, **kwargs)
    else:
        return json.load(fp, object_hook=_json_deserialize, **kwargs)

def loads_custom(s, encoding=None, **kwargs):
    return json.loads(s, encoding=encoding, object_hook=_json_deserialize, **kwargs)

def dump_custom(obj, fp, **kwargs):
    if isinstance(fp, str):
        with open(fp, 'w') as f:
            return json.dump(obj, f, default=_json_serialize, **kwargs)
    else:
        return json.dump(obj, fp, default=_json_serialize, **kwargs)

def dumps_custom(obj, **kwargs):
    return json.dumps(obj, default=_json_serialize, **kwargs)