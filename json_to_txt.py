
import csv
import json

if __name__ == '__main__':
    with open('thesaurus.json') as f:
        data = json.load(f)

    with open('json_thesaurus.txt', 'w') as f:
        writer = csv.writer(f)
        for word, syns in data.items():
            row = [w.replace(' ', '_') for w in [word] + syns]
            writer.writerow(row, delimiter=' ')
