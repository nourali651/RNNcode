# === 1. Define helper math functions ===
def exp(x):
    n = 20  # number of terms in Taylor series
    result = 1.0
    term = 1.0
    for i in range(1, n):
        term *= x / i
        result += term
    return result

def tanh(x):
    e_pos = exp(x)
    e_neg = exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

def log(x):
    n = 100
    result = 0
    y = (x - 1) / (x + 1)
    y2 = y * y
    for i in range(1, n, 2):
        result += (1/i) * (y ** i)
    return 2 * result

# === 2. Data Preparation ===
text = ["dogs", "giraffe", "cats", "zebra"]
vocab = list(set(text))
vocab_size = len(vocab)

word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for i, w in enumerate(vocab)}

def one_hot(ix, size):
    vec = [0 for _ in range(size)]
    vec[ix] = 1
    return vec

# === 3. Initialize Weights manually (fixed small values) ===
input_size = vocab_size
hidden_size = 5
output_size = vocab_size

def rand():
    rand.seed = (rand.seed * 1103515245 + 12345) % (2**31)
    return ((rand.seed >> 16) & 0x7FFF) / 32767

rand.seed = 42

Wxh = [[(rand() - 0.5) * 0.2 for _ in range(hidden_size)] for _ in range(input_size)]
Whh = [[(rand() - 0.5) * 0.2 for _ in range(hidden_size)] for _ in range(hidden_size)]
Why = [[(rand() - 0.5) * 0.2 for _ in range(output_size)] for _ in range(hidden_size)]

bh = [0 for _ in range(hidden_size)]
by = [0 for _ in range(output_size)]

# === 4. Helper Functions ===
def matmul(vec, mat):
    return [sum(vec[j] * mat[j][i] for j in range(len(vec))) for i in range(len(mat[0]))]

def add(vec1, vec2):
    return [vec1[i] + vec2[i] for i in range(len(vec1))]

def softmax(vec):
    exps = [exp(v) for v in vec]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def cross_entropy(pred, target_ix):
    return -log(pred[target_ix] + 1e-9)

def argmax(vec):
    return vec.index(max(vec))

def tanh_vec(vec):
    return [tanh(v) for v in vec]

def matTmul(vec, mat):
    return [sum(vec[j] * mat[i][j] for j in range(len(vec))) for i in range(len(mat))]

# === 5. Training Loop ===
learning_rate = 0.1

for epoch in range(500):
    h = [0 for _ in range(hidden_size)]
    inputs = text[:3]
    target = word_to_ix[text[3]]
    xs, hs = [], [h]

    for word in inputs:
        x = one_hot(word_to_ix[word], vocab_size)
        xs.append(x)
        h = tanh_vec(add(add(matmul(x, Wxh), matmul(h, Whh)), bh))
        hs.append(h)

    y = add(matmul(h, Why), by)
    p = softmax(y)

    loss = cross_entropy(p, target)

    dWhy = [[0 for _ in range(output_size)] for _ in range(hidden_size)]
    dWhh = [[0 for _ in range(hidden_size)] for _ in range(hidden_size)]
    dWxh = [[0 for _ in range(hidden_size)] for _ in range(input_size)]
    dbh = [0 for _ in range(hidden_size)]
    dby = [0 for _ in range(output_size)]

    dh_next = [0 for _ in range(hidden_size)]

    dy = p[:]
    dy[target] -= 1

    for i in range(hidden_size):
        for j in range(output_size):
            dWhy[i][j] += hs[-1][i] * dy[j]
    for j in range(output_size):
        dby[j] += dy[j]

    dh = matTmul(dy, Why)
    dh = add(dh, dh_next)

    for t in range(2, -1, -1):
        h_raw = hs[t+1]
        dh_raw = [(1 - h_raw[i] ** 2) * dh[i] for i in range(hidden_size)]

        for i in range(hidden_size):
            for j in range(hidden_size):
                dWhh[j][i] += hs[t][j] * dh_raw[i]
            for j in range(input_size):
                dWxh[j][i] += xs[t][j] * dh_raw[i]
            dbh[i] += dh_raw[i]

        dh = matTmul(dh_raw, Whh)

    for i in range(input_size):
        for j in range(hidden_size):
            Wxh[i][j] -= learning_rate * dWxh[i][j]
    for i in range(hidden_size):
        for j in range(hidden_size):
            Whh[i][j] -= learning_rate * dWhh[i][j]
    for i in range(hidden_size):
        for j in range(output_size):
            Why[i][j] -= learning_rate * dWhy[i][j]
    for i in range(hidden_size):
        bh[i] -= learning_rate * dbh[i]
    for i in range(output_size):
        by[i] -= learning_rate * dby[i]

    if epoch % 50 == 0:
        print("Epoch", epoch, "loss:", loss)

# === 6. Inference ===
print("\nAfter training:")

h = [0 for _ in range(hidden_size)]
for word in text[:3]:
    x = one_hot(word_to_ix[word], vocab_size)
    h = tanh_vec(add(add(matmul(x, Wxh), matmul(h, Whh)), bh))

y = add(matmul(h, Why), by)
p = softmax(y)
predicted_ix = argmax(p)

print("Predicted 4th word:", ix_to_word[predicted_ix])
print("Actual 4th word:", text[3])
