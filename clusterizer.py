import heapq
import itertools
import re
import sys


def remove_special_chars(text):
    return re.sub('[^\w]+', '', text.lower())


def generate_ngrams(text, n):
    split_ngrams = zip(*[text[i:] for i in range(n)])
    ngrams = map(lambda x: ''.join(x), split_ngrams)
    return frozenset(ngrams)


def read_data(file_path):
    with open(file_path) as f:
        return map(str.strip, f.readlines())


def calculate_dice_distance(x, y, ngrams):
    x = hash(x)
    y = hash(y)
    # print(ngrams[x], ngrams[y])
    return 1.0 - (2.0 * float(len(ngrams[x] & ngrams[y])) / float(len(ngrams[x]) + len(ngrams[y])))


def prepare_ngrams(data, n):
    ngrams = {}
    for e in data:
        ngrams[hash(e)] = generate_ngrams(e, n)
    return ngrams


def preprocess_data(data):
    data_registry = {}
    processed_data = []
    for e in data:
        processed = remove_special_chars(e)
        processed_data.append(processed)
        data_registry[hash(processed)] = e
    return processed_data, data_registry


def calculate_cluster_distance(a, b, ngrams):
    return min([calculate_dice_distance(x, y, ngrams) for x in a for y in b])


def calculate_cluster_average(cluster, ngrams):
    distances = []
    for a, b in itertools.combinations(cluster, 2):
        distances.append(calculate_dice_distance(a, b, ngrams))
    if len(distances):
        return sum(distances) / len(distances)
    else:
        return 0.5


def calculate_level_average(level, ngrams):
    distances = list(map(lambda x: calculate_cluster_average(x, ngrams), level))
    return sum(distances) / len(distances)


def generate_clusters(data, ngrams):
    clusters = [frozenset([e]) for e in data]
    cluster_registry = {hash(c): c for c in clusters}
    size = len(clusters)
    # dendrogram = [clusters[:]]
    dendrogram = clusters[:]
    min_avg_distance = 0.5
    heap = []
    for c1, c2 in itertools.combinations(clusters, 2):
        distance = calculate_cluster_distance(c1, c2, ngrams)
        heapq.heappush(heap, (distance, hash(c1), hash(c2)))
    while len(heap):
        d, ch1, ch2 = heapq.heappop(heap)
        c1, c2 = cluster_registry[ch1], cluster_registry[ch2]
        if not c1 in clusters:
            continue
        clusters.remove(c1)
        if not c2 in clusters:
            continue
        clusters.remove(c2)
        c3 = c1 | c2
        ch3 = hash(c3)
        if cluster_registry.get(ch3, None):
            continue
        cluster_registry[ch3] = c3
        dl = clusters[:]
        dl.append(c3)
        avg_dist = calculate_level_average(dl, ngrams)
        # dendrogram.append(dl)
        if avg_dist < min_avg_distance:
            min_avg_distance = avg_dist
            dendrogram = dl
        for c in clusters:
            distance = calculate_cluster_distance(c3, c, ngrams)
            heapq.heappush(heap, (distance, ch3, hash(c)))
        clusters.append(c3)
    return min_avg_distance, dendrogram


def main():
    file_path = sys.argv[1]
    n = int(sys.argv[2])
    data = read_data(file_path)
    processed_data, data_registry = preprocess_data(data)
    ngrams = prepare_ngrams(processed_data, n)
    min_avg_dist, dendrogram = generate_clusters(processed_data, ngrams)
    print(min_avg_dist)
    for c in dendrogram:
        for e in c:
            print(data_registry[hash(e)])
        print('\n')


if __name__ == '__main__':
    main()
