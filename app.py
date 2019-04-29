from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range



class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)


#Data Generation
def datageneration_add(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE):
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a + b)
        ans += ' ' * (DIGITS + 1 - len(ans))
        if REVERSE:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    print(questions[:5], expected[:5])

    return questions,expected

def datageneration_sub(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE):
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        a,b = sorted((a, b))
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}-{}'.format(b, a)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(b - a)
        ans += ' ' * (DIGITS + 1 - len(ans))
        if REVERSE:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    print(questions[:5], expected[:5])

    return questions,expected

def datageneration_both(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE):
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        if(len(questions)%2 == 0):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            q = '{}+{}'.format(a, b)
            query = q + ' ' * (MAXLEN - len(q))
            ans = str(a + b)
            ans += ' ' * (DIGITS + 1 - len(ans))
            if REVERSE:
                query = query[::-1]
            questions.append(query)
            expected.append(ans)
        else:
            b, a = sorted((a, b))
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            q = '{}-{}'.format(a, b)
            query = q + ' ' * (MAXLEN - len(q))
            ans = str(a - b)
            ans += ' ' * (DIGITS + 1 - len(ans))
            if REVERSE:
                query = query[::-1]
            questions.append(query)
            expected.append(ans)

    print('Total addition questions:', len(questions))

    print(questions[:5], expected[:5])

    return questions,expected

def datageneration_mul(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE):
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}*{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a * b)
        ans += ' ' * (DIGITS*2  - len(ans))
        if REVERSE:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    print(questions[:5], expected[:5])

    return questions,expected

#Processing
def Vectorization_split(questions,expected,DIGITS,MAXLEN,chars):
    print('Vectorization...')
    ctable = CharacterTable(chars)
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x[:20000]
    train_y = y[:20000]
    test_x = x[20000:]
    test_y = y[20000:]

    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)

    print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])
    
    return ctable,x_train,y_train,x_val,y_val,test_x,test_y

def Vectorization_split2(questions,expected,DIGITS,MAXLEN,chars):
    print('Vectorization...')
    ctable = CharacterTable(chars)
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), DIGITS*2, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS*2)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x[:20000]
    train_y = y[:20000]
    test_x = x[20000:]
    test_y = y[20000:]

    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)

    print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])
    
    return ctable,x_train,y_train,x_val,y_val,test_x,test_y
#Build MODEL
def buildmodel(seq_len,vec_dim):
    print('Build model...')
    model = Sequential()
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.core.Dense(128, activation="relu"))
    model.add(layers.core.RepeatVector(seq_len))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.wrappers.TimeDistributed(layers.core.Dense(vec_dim, activation="softmax"))) 
    model.compile(loss="categorical_crossentropy", optimizer='adam')
    #model.summary()
    return model

#Training
def train(model,x_train,y_train,x_val,y_val,ctable,BATCH_SIZE,REVERSE):
    for iteration in range(100):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x_train, y_train,batch_size=BATCH_SIZE,epochs=1,validation_data=(x_val, y_val))
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if REVERSE else q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print('☑' , end=' ')
            else:
                print('☒' , end=' ')
            print(guess)
    return model
#Testing
def testing(model,ctable,test_x,test_y):
    print("MSG : Prediction")
    preds = model.predict_classes(test_x, verbose=0)
    count = 0
    n = len(preds)
    for i in range(n):
        #q = ctable.decode(test_x[i])
        correct = ctable.decode(test_y[i])
        guess = ctable.decode(preds[i], calc_argmax=False)
        if(correct == guess):
            count += 1
    print(count,n)
    print('acc:', count/n)



def adder():
    #Parameters Config
    TRAINING_SIZE = 80000
    DIGITS = 3
    REVERSE = False
    MAXLEN = DIGITS + 1 + DIGITS
    chars = '0123456789+ '
    BATCH_SIZE = 128
    #Data Generation
    questions,expected = datageneration_add(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE)
    #Processing
    ctable,x_train,y_train,x_val,y_val,test_x,test_y = Vectorization_split(questions,expected,DIGITS,MAXLEN,chars)
    #Build MODEL
    model = buildmodel(DIGITS+1,len(chars))
    #Training
    model = train(model,x_train,y_train,x_val,y_val,ctable,BATCH_SIZE,REVERSE)
    #Testing
    testing(model,ctable,test_x,test_y)


def sub():
    #Parameters Config
    TRAINING_SIZE = 80000
    DIGITS = 3
    REVERSE = False
    MAXLEN = DIGITS + 1 + DIGITS
    chars = '0123456789- '
    BATCH_SIZE = 128
    #Data Generation
    questions,expected = datageneration_sub(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE)
    #Processing
    ctable,x_train,y_train,x_val,y_val,test_x,test_y = Vectorization_split(questions,expected,DIGITS,MAXLEN,chars)
    #Build MODEL
    model = buildmodel(DIGITS+1,len(chars))
    #Training
    model = train(model,x_train,y_train,x_val,y_val,ctable,BATCH_SIZE,REVERSE)
    #Testing
    testing(model,ctable,test_x,test_y)

def combine():
    #Parameters Config
    TRAINING_SIZE = 80000
    DIGITS = 3
    REVERSE = False
    MAXLEN = DIGITS + 1 + DIGITS
    chars = '0123456789+- '
    BATCH_SIZE = 128
    #Data Generation
    questions,expected = datageneration_both(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE)
    #Processing
    ctable,x_train,y_train,x_val,y_val,test_x,test_y = Vectorization_split(questions,expected,DIGITS,MAXLEN,chars)
    #Build MODEL
    model = buildmodel(DIGITS+1,len(chars))
    #Training
    model = train(model,x_train,y_train,x_val,y_val,ctable,BATCH_SIZE,REVERSE)
    #Testing
    testing(model,ctable,test_x,test_y)

def mul():
    #Parameters Config
    TRAINING_SIZE = 80000
    DIGITS = 3
    REVERSE = False
    MAXLEN = DIGITS + 1 + DIGITS
    chars = '0123456789* '
    BATCH_SIZE = 128
    #Data Generation
    questions,expected = datageneration_mul(TRAINING_SIZE,DIGITS,MAXLEN,REVERSE)
    #Processing
    ctable,x_train,y_train,x_val,y_val,test_x,test_y = Vectorization_split2(questions,expected,DIGITS,MAXLEN,chars)
    #Build MODEL
    model = buildmodel(DIGITS*2,len(chars))
    #Training
    model = train(model,x_train,y_train,x_val,y_val,ctable,BATCH_SIZE,REVERSE)
    #Testing
    testing(model,ctable,test_x,test_y)


def main():
    adder()
    sub()
    combine()
    #mul()

    
if __name__ == '__main__':
    main()