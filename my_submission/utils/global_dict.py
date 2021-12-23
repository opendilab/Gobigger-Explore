def GlobalDictInit():
    global _global_dict
    _global_dict = {}


def SetValueDict(key, value):
    if key in _global_dict.keys():
        _global_dict[key]+=[value]
    else:
        _global_dict[key] = []
        _global_dict[key]+=[value]


def GetValueDict(key):
    try:
        return _global_dict[key]
    except:
        print('Access' + key + 'error\r\n')