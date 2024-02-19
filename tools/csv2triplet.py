import json

def csv2triples(csv, separator='\t', delimiter='\n'):  
    lines = csv.strip().split(delimiter)
    header = lines[0].split(separator)
    triples = []
    for line in lines[1:]:
        if not line:
            continue
        values = line.split(separator)
        entity = values[0]
        for i in range(1, len(values)):
            if i >= len(header):
                break
            #triples.append((entity, header[i], values[i]))
            #---------------------------------------------------------
            temp = [entity, header[i]]  
            triples.append('(' + temp[0].strip() + ',' + temp[1].strip() + ',' + values[i].strip() + ')')
            #---------------------------------------------------------
    return triples

