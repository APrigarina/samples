import cv2
import numpy as np
import sys
import time
from itertools import islice
from shapely.geometry import Polygon
import optparse
import os
import sys
from os.path import join
from shapely.geometry import Polygon


# Handle command line options
p = optparse.OptionParser()
p.add_option('--Images', '-i', default="qrcodes/detection",help="Location of directory with input images")
p.add_option('--Results', '-r', default="detection_results",help="Location of root results directory")

def parse_truth( file_path ):
    locations = []
    with open(file_path) as f:
        sets = False
        corners = []
        for line in f:
            if line[0] is '#':
                continue
            if line.startswith("SETS"):
                sets = True
                continue
            values = [float(s) for s in line.split()]
            if sets:
                if len(values) != 8:
                    print("Expected 4 corners in truth. "+file_path)
                    print(values)
                    exit(1)
                else:
                    locations.append(values)
            else:
                corners.extend(values)

        if not sets:
            if len(corners) != 8:
                print("Expected 4 corners in truth. "+file_path)
                print(corners)
                exit(1)
            else:
                locations.append(corners)
    return locations

def parse_results( file_path ):
    locations = []
    loc = []
    milliseconds = 0
    with open(file_path) as f:
        for line in f:
            # Skip comments and messages
            if line[0] is '#' or line.startswith("message"):
                continue
            if line.startswith("milliseconds"):
                milliseconds = float(line.split()[2])
            else:
                values = [float(s) for s in line.split()]
                if len(values) != 8:
                    print("Expected 4 corners in results. "+file_path)
                    print(values)
                    exit(1)
                else:
                    locations.append(values)
    return {'locs':locations,'ms':milliseconds}

def reshape_list( l ):
    output = []
    for i in range(0,len(l),2):
        output.append((l[i],l[i+1]))
    return output

def compare_location_results(expected, found):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    ambiguous = 0

    paired = [False]*len(found)

    for e in expected:
        p1=Polygon(reshape_list(e))
        total_matched = 0
        for idx,f in enumerate(found):
            p2=Polygon(reshape_list(f))
            # print("p2", p2)
            try:
                x = p1.intersection(p2)
                y = p1.union(p2)
                if x.area/y.area > 0.5:
                    paired[idx] = True
                    total_matched += 1
                    true_positive += 1
                # print("paired", paired)
            except:
                pass # not sure what to do here
        if total_matched == 0:
            false_negative += 1
        elif total_matched > 1:
            ambiguous += 1

    for idx in range(len(found)):
        if not paired[idx]:
            false_positive += 1

    return {"tp":true_positive,"fp":false_positive,"fn":false_negative,"ambiguous":ambiguous}

def compute_f( tp , fp , tn , fn ):
    if tp == 0 and fp == 0 and fn == 0:
        return 0
    return 2.0*tp/(2.0*tp + fp + fn)

options, arguments = p.parse_args()

dir_images = options.Images
dir_results = options.Results

#print(dir_images) #'brightness', 'damaged',
#print(dir_results) #results

# Number of images in each category is stored here
category_counts = {}

# List of all the datasets
data_sets = [d for d in os.listdir(dir_images) if os.path.isdir(join(dir_images,d))] #'brightness', 'damaged', ...
print(data_sets)


# name of directories in the root directory is the same as the project which generated them
#for target_name in sorted(os.listdir(dir_results)):
#    if not os.path.isdir(join(dir_results,target_name)):
#        continue

    #path_to_target = os.path.join(dir_results,target_name) #results/blurred
    #print("path_to_target", path_to_target)
total_missing = 0
total_false_positive = 0
total_false_negative = 0
total_true_positive = 0
total_ambiguous = 0

ds_results = {}

for ds in sorted(data_sets):
    path_ds_results = join(dir_results, ds)
    #print("path_ds_results",path_ds_results)
    path_ds_truth = join(dir_images,ds)
    #print("path_ds_truth", path_ds_truth)
    truth_files = [f for f in os.listdir(path_ds_truth) if f.endswith("txt")]
    #print("truth files", truth_files)
    category_counts[ds] = len(truth_files)

    ds_true_positive = 0
    ds_false_positive = 0
    ds_false_negative = 0

    milliseconds = []

    for truth_file in truth_files:
        if not os.path.isfile(join(path_ds_results,truth_file)):
            total_missing += 1
            print("Missing results for {}".format(join(ds,truth_file)))
            continue

        expected = parse_truth(join(path_ds_truth,truth_file))
        # print("expected", expected)
        # print(join(path_ds_truth,truth_file))
        try:
            found = parse_results(join(path_ds_results,truth_file))
            # print("path", path_ds_results, truth_file)
            # print("found", found['locs'])
        except Exception as e:
            print("Failed parsing {} {}".format(path_ds_results,truth_file))
            print("error = {}".format(e))
            raise e
        metrics_loc = compare_location_results(expected, found['locs'])

        milliseconds.append(found['ms'])

        total_ambiguous += metrics_loc['ambiguous']
        total_false_positive += metrics_loc['fp']
        total_true_positive += metrics_loc['tp']
        total_false_negative += metrics_loc['fn']

        ds_false_negative += metrics_loc['fn']
        ds_true_positive += metrics_loc['tp']
        ds_false_positive += metrics_loc['fp']

    #milliseconds.sort()
    #ms50 = milliseconds[int(len(milliseconds)/2)]
    #ms95 = milliseconds[int(len(milliseconds)*0.95)]
    #msMean = sum( milliseconds) / len(milliseconds)

    ds_results[ds] = {"tp":ds_true_positive,"fp":ds_false_positive,"fn":ds_false_negative}


print()
print("=============== {} ================".format("detection"))
print("  total input      {}".format(total_true_positive+total_false_negative))
print("  missing results  {}".format(total_missing))
print()
print("  false positive   {}".format(total_false_positive))
print("  false negative   {}".format(total_false_negative))
print("  true positive    {}".format(total_true_positive))
print("  ambiguous        {}".format(total_ambiguous))
print()
for k, v in ds_results.items():
    print(k, v['tp'] ,v['fn'])
scoresF = {}
scoresRun = {}

scoresF["summary"] = compute_f(total_true_positive,total_false_positive,0,total_false_negative)
#scoresRun["summary"] = sum( [ds_results[n]["ms50"] for n in ds_results]) / len(ds_results)
for n in sorted(list(ds_results.keys())):

    r = ds_results[n]

    F = compute_f(r['tp'],r['fp'],0,r['fn'])
    scoresF[n] = F
print('\nF-score ')
for key, value in scoresF.items():
    print(key, value)
f_values = list(scoresF.values())[1:]
print()
print("Mean F-score:", np.mean(f_values))
        #scoresRun[n] = {"ms50":r["ms50"], "ms95":r["ms95"], "msMean":r["msMean"]}
        #print("{:15s} F={:.2f} ms50={:7.2f} ms95={:7.2f}".format(n,F,r["ms50"],r["ms95"]))


# Create the plot showing a summary by category
import matplotlib.pyplot as plt
import numpy as np
categories = sorted(list(scoresF.keys()))
categories = [x for x in categories if x is not "summary"]
scores_list = [scoresF[x] for x in categories]

indexes = np.arange(len(categories))
fig, ax = plt.subplots()

ax.bar(indexes,scores_list)

ax.set_ylim([0.0,1.15])
ax.set_ylabel("F-Score (1 = best)")
ax.set_title('Detection Performance by Categories')
ax.set_xticks(indexes)
ax.set_xticklabels(categories, rotation=90)


plt.gcf().subplots_adjust(bottom=0.25)
fig.set_size_inches(12, 4)
plt.savefig("detection_f_score_by_categories.pdf", format='pdf')
plt.close()
