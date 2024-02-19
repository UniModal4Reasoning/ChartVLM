import re
import string
from typing import Any, Callable, Optional, Sequence
import datasets
import numpy as np
import Levenshtein



def csv_eval(predictions,references,easy):
    predictions = np.asarray(predictions)
    labels = np.asarray(references)
    def is_int(val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    def is_float(val):
        try:
            float(val)
            return True
        except ValueError:
            return False
    
    def csv2triples(csv, separator='\\t', delimiter='\\n'):  
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
                temp = sorted([entity.strip(), header[i].strip()])  
                value = values[i].strip()
                value = value.replace("%","")     
                value = value.replace("$","")     
                triples.append((temp[0].strip(), temp[1].strip(), value))
                #---------------------------------------------------------
        return triples
    
    def csv2triples_plotqa(csv, separator='\\t', delimiter='\\n'):  
        lines = csv.strip().split(delimiter)
        header = lines[1].split(separator) 
        triples = []
        for line in lines[2:]:   
            if not line:
                continue
            values = line.split(separator)
            entity = values[0]
            for i in range(1, len(values)):
                if i >= len(header):
                    break
                #triples.append((entity, header[i], values[i]))
                #---------------------------------------------------------
                temp = sorted([entity.strip(), header[i].strip()])  
                value = values[i].strip()
                value = value.replace("%","")     
                value = value.replace("$","")     
                triples.append((temp[0].strip(), temp[1].strip(), value))
                #---------------------------------------------------------
        return triples

    def process_triplets(triplets):
        new_triplets = []
        for triplet in triplets:
            new_triplet = []
            triplet_temp = []
            if len(triplet) > 2:
                if is_int(triplet[2]) or is_float(triplet[2]):
                    triplet_temp = (triplet[0].lower(), triplet[1].lower(), float(triplet[2]))
                else:
                    triplet_temp = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
            else: 
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), "no meaning")
            new_triplets.append(triplet_temp)
        return new_triplets

    def intersection_with_tolerance(a, b, tol_word, tol_num):
        a = set(a)
        b = set(b)
        c = set()
        for elem1 in a:
            for elem2 in b:
                if is_float(elem1[-1]) and is_float(elem2[-1]):
                    if (Levenshtein.distance(''.join(elem1[:-1]),''.join(elem2[:-1])) <= tol_word) and (abs(elem1[-1] - elem2[-1]) / (elem2[-1]+0.000001) <= tol_num):
                        c.add(elem1)
                else:
                    if (Levenshtein.distance(''.join([str(i) for i in elem1]),''.join([str(j) for j in elem2])) <= tol_word):
                        c.add(elem1)
        return list(c)

    def union_with_tolerance(a, b, tol_word, tol_num):
        c = set(a) | set(b)
        d = set(a) & set(b)
        e = intersection_with_tolerance(a, b, tol_word, tol_num)
        f = set(e)
        g = c-(f-d)
        return list(g)

    def get_eval_list(pred_csv, label_csv, separator='\\t', delimiter='\\n', tol_word=3, tol_num=0.05):
        pred_triple_list=[]
        for it in pred_csv:
            pred_triple_temp = csv2triples(it, separator=separator, delimiter=delimiter)
            #pred_triple_temp = csv2triples_plotqa(it, separator=separator, delimiter=delimiter)  #plotqa
            pred_triple_pre = process_triplets(pred_triple_temp)
            pred_triple_list.append(pred_triple_pre) 
        #print(pred_triple_list[0])

        label_triple_list=[]
        for it in label_csv:
            label_triple_temp = csv2triples(it, separator='\\t', delimiter='\\n')
            label_triple_pre = process_triplets(label_triple_temp)
            label_triple_list.append(label_triple_pre) 
        #print(label_triple_list[0])

        intersection_list=[]
        union_list=[]
        sim_list=[]
        for pred,label in zip(pred_triple_list, label_triple_list):
            intersection = intersection_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
            union = union_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
            sim = len(intersection)/len(union)
            intersection_list.append(intersection)
            union_list.append(union)
            sim_list.append(sim)
        return intersection_list, union_list, sim_list

    def get_ap(predictions, labels, sim_threhold, tolerance, separator='\\t', delimiter='\\n', easy=1):
        if tolerance == 'strict':
            tol_word=0
            if easy == 1:
                tol_num=0
            else:
                tol_num=0.1

        elif tolerance == 'slight':
            tol_word=2
            if easy == 1:
                tol_num=0.05
            else:
                tol_num=0.3

        elif tolerance == 'high':
            tol_word= 5
            if easy == 1:
                tol_num=0.1
            else:
                tol_num=0.5      
        intersection_list, union_list, sim_list = get_eval_list(predictions, labels, separator=separator, delimiter=delimiter, tol_word=tol_word, tol_num=tol_num)
        ap = len([num for num in sim_list if num >= sim_threhold])/len(sim_list)
        return ap   

    map_strict = 0
    map_slight = 0
    map_high = 0
    s="\\t"
    d="\\n"

    for sim_threhold in np.arange (0.5, 1, 0.05):
        map_temp_strict = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='strict', separator=s, delimiter=d, easy=easy)
        map_temp_slight = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='slight', separator=s, delimiter=d, easy=easy)
        map_temp_high = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='high', separator=s, delimiter=d, easy=easy)
        map_strict += map_temp_strict/10
        map_slight += map_temp_slight/10
        map_high += map_temp_high/10

    em = get_ap(predictions, labels, sim_threhold=1, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_50_strict = get_ap(predictions, labels, sim_threhold=0.5, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_75_strict = get_ap(predictions, labels, sim_threhold=0.75, tolerance='strict', separator=s, delimiter=d, easy=easy)    
    ap_90_strict = get_ap(predictions, labels, sim_threhold=0.90, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_50_slight = get_ap(predictions, labels, sim_threhold=0.5, tolerance='slight', separator=s, delimiter=d, easy=easy)
    ap_75_slight = get_ap(predictions, labels, sim_threhold=0.75, tolerance='slight', separator=s, delimiter=d, easy=easy)    
    ap_90_slight = get_ap(predictions, labels, sim_threhold=0.90, tolerance='slight', separator=s, delimiter=d, easy=easy)
    ap_50_high = get_ap(predictions, labels, sim_threhold=0.5, tolerance='high', separator=s, delimiter=d, easy=easy)
    ap_75_high = get_ap(predictions, labels, sim_threhold=0.75, tolerance='high', separator=s, delimiter=d, easy=easy)    
    ap_90_high = get_ap(predictions, labels, sim_threhold=0.90, tolerance='high', separator=s, delimiter=d, easy=easy)


    return em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high

def draw_SCRM_table(em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high):

    result=f'''
            -----------------------------------------------------------\n
            |  Metrics   |  Sim_threshold  |  Tolerance  |    Value    |\n
            -----------------------------------------------------------\n
            |            |                 |   strict    |    {'%.4f' % map_strict}    |     \n
            |            |                 ----------------------------\n
            | mPrecison  |  0.5:0.05:0.95  |   slight    |    {'%.4f' % map_slight}    |\n
            |            |                  ---------------------------\n
            |            |                 |    high     |    {'%.4f' % map_high}    |\n
            -----------------------------------------------------------\n
            |            |                 |   strict    |    {'%.4f' % ap_50_strict}    |\n
            |            |                  ---------------------------\n
            | Precison   |       0.5       |   slight    |    {'%.4f' % ap_50_slight }    |\n
            |            |                  ---------------------------\n
            |            |                 |    high     |    {'%.4f' % ap_50_high }    |\n
            -----------------------------------------------------------\n
            |            |                 |   strict    |    {'%.4f' % ap_75_strict}    |\n
            |            |                  ---------------------------\n
            | Precison   |      0.75       |   slight    |    {'%.4f' % ap_75_slight}    |\n
            |            |                  ---------------------------\n
            |            |                 |    high     |    {'%.4f' % ap_75_high}    |\n
            -----------------------------------------------------------\n
            |            |                 |   strict    |    {'%.4f' % ap_90_strict}    |\n
            |            |                  ---------------------------\n
            | Precison   |       0.9       |   slight    |    {'%.4f' % ap_90_slight }    |\n
            |            |                  ---------------------------\n
            |            |                 |    high     |    {'%.4f' % ap_90_high}    |\n
            -----------------------------------------------------------\n
            |Precison(EM)|        1        |   strict    |    {'%.4f' % em}    |\n
            -----------------------------------------------------------\n
            '''
    return result