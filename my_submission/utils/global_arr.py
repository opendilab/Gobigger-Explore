


from easydict import EasyDict
   
def GlobalArrInit():
    global _global_arr
    _global_arr = []


def SetValueArr(value):
    _global_arr.append(value)


def GetValueArr():
    return _global_arr