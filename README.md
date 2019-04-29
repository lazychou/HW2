class CharacterTable用於表示向量的轉換

函數datageneration_add用於產生加法器的資料
函數datageneration_sub用於產生減法器的資料
函數datageneration_combine用於產生加減法器的資料
函數datageneration_mul分別用於產生乘法器的資料

函數Vectorization_split用於加法器、減法器以及加減法器的資料分割，target的序列長度為digit+1
函數Vectorization_split用於乘法器的資料分割，target的序列長度為digit*2

函數buildmodel建立模型，加法器、減法器、加減法器以及乘法器都使用同一種架構的模型

函數train則是訓練模型，並且會在每個epoch印出部分預測結果

函數testing用訓練好的模型預測預測資料，並顯示準確率

函數adder為加法器的程式
函數sub為減法器的程式
函數combine為加減法器的程式
函數mul為乘法器的程式

實驗的參數:
RAINING_SIZE = 80000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
BATCH_SIZE = 128
epoch = 100

結果:
加法器和減法器都能達到98%以上的準確率
加減法器則是下降到30%左右
而乘法器不到1%，我的模型設計不適合乘法器。
