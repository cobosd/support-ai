def getValues(list, key):
    values = []
    for obj in list:
        if str(key) in obj:
            values.append(obj[str(key)])
    return values