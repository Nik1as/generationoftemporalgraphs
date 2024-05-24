import random
import math
import heapq
from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def generate(self):
        pass


class Uniform(Distribution):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def generate(self):
        return random.randint(self.start, self.end)


class Alias(Distribution):
    def __init__(self, p):
        self.n = len(p)
        self.prob = [0] * self.n
        self.alias = [0] * self.n

        small = []
        large = []

        for i in range(self.n):
            p[i] = p[i] * self.n
            if p[i] < 1:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            i = small.pop()
            j = large.pop()
            self.prob[i] = p[i]
            self.alias[i] = j
            p[j] = (p[j] + p[i]) - 1
            if p[j] < 1:
                small.append(j)
            else:
                large.append(j)
        while large:
            i = large.pop()
            self.prob[i] = 1
        while small:
            i = small.pop()
            self.prob[i] = 1

    def generate(self):
        i = random.randrange(self.n)
        if random.random() < self.prob[i]:
            return i
        else:
            return self.alias[i]


class AliasAddOne(Alias):

    def __init__(self, p):
        super().__init__(p)

    def generate(self):
        return super().generate() + 1


class Zipf(Alias):
    def __init__(self, gamma, n):
        c = 1 / sum(1 / k**gamma for k in range(1, n + 1))
        super().__init__([c * x ** (-gamma) for x in range(1, n + 1)])

    def generate(self):
        return super().generate() + 1


class WeightedSamplingWithoutReplacement(ABC):
    
    def __init__(self, population, weights):
        self.population = population
        self.weights = weights

    @abstractmethod
    def sample(self, k):
        pass


class RejectionSampling(WeightedSamplingWithoutReplacement):

    def __init__(self, population, weights):
        super().__init__(population, weights)
        weights_sum = sum(weights)
        self.weights = [w / weights_sum for w in self.weights]
        self.alias = Alias(self.weights)

    
    def sample(self, k):
        result = set()
        while len(result) < k:
            result.add(self.alias.generate())
        return [self.population[i] for i in result]


class EfraimidisSpirakis(WeightedSamplingWithoutReplacement):

    def __init__(self, population, weights):
        super().__init__(population, weights)

    def sample(self, k):
        keys = [random.random() ** (1 / w) for w in self.weights]
        indices = heapq.nlargest(k, range(len(self.population)), key=lambda item: keys[item])
        return [self.population[i] for i in indices]


class Hybrid(WeightedSamplingWithoutReplacement):
    def __init__(self, population, weights, threshold=None):
        super().__init__(population, weights)
        self.rejection = RejectionSampling(population, weights)
        self.efraimidis_spirakis = EfraimidisSpirakis(population, weights)
        self.threshold = threshold or len(population) / 4

    def sample(self, k):
        if k < self.threshold:
            return self.rejection.sample(k)
        else:
            return self.efraimidis_spirakis.sample(k)


def generate_burst_weights(waves, k):
    return [(k**x / math.factorial(x)) * math.exp(-k) for x in range(2 * k)] * waves
