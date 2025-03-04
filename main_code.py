import pandas as pd # For I/O, Data Transformation
from sklearn.model_selection import train_test_split
# install scikit learn
# install nltk
# install sastrawi

## Import data ke dataframe
data1 = pd.read_excel("archive/Summarized/dataset_cnn_summarized.xlsx")
data2 = pd.read_excel("archive/Summarized/dataset_kompas_summarized.xlsx")
data3 = pd.read_excel("archive/Summarized/dataset_tempo_summarized.xlsx")
data4 = pd.read_excel("archive/Summarized/dataset_turnbackhoax_summarized.xlsx")

df1 = data1[['summarized','label']]
df2 = data2[['summarized','label']]
df3 = data3[['summarized','label']]
df4 = data4[['summarized','label']]

df = pd.concat([df1,df2,df3,df4])
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index() 
df = df.iloc[:,1:]
label = df.label.value_counts()

df = df.sort_values(by='label')
df = df.reset_index()
#  
df = df.iloc[:,1:]
print(df.head(10),"\n")
print(df.groupby('label').count(),"\n")

### CASE FOLDING ###
print('\nCASE FOLDING\n')
def case_folding(data):
    import re
    # Lowertext (tidak kapital)
    data = data.lower()
    # Menghilangkan Angka 0-9, Tanda Baca, Karakter Spesial, dan Emoji
    data = re.sub(r'[^A-Za-z]', ' ', data)
    # Menghilangkan Whitespace
    data = data.strip()
    return data

df['case_folding'] = df['summarized'].apply(lambda x : case_folding(x))

### TOKENIZING ###
print('\nTOKENIZING\n')
def tokenize(teks):
    import nltk
    nltk.download('punkt_tab')
    teks = nltk.tokenize.word_tokenize(teks)
    return teks
df['tokenizing'] = df['case_folding'].apply(lambda x : tokenize(x))

### REMOVE STOPWORD ###
print('\nSTOPWORD\n')
def rmstopwords(teks):
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    swsastra = factory.get_stop_words()
    nostopwords = []
    for word in teks:
        if (word not in swsastra):
            nostopwords.append(word)
    return nostopwords
df['stopwords_removed'] = df['tokenizing'].apply(lambda x : rmstopwords(x))

### STEMMING ###
print('\nSTEMMING\n')

import time # digunakan untuk menghitung waktu compile
tm = time.ctime()
print("Waktu program dimulai :",tm)
start = time.time()

def stemming(teks):
    cleantext = []
    for word in teks:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factorystem = StemmerFactory()
        stemmer = factorystem.create_stemmer()
        stemmed_word = stemmer.stem(word)
        cleantext.append(stemmed_word)
    return cleantext
df['stemming'] = df['stopwords_removed'].apply(lambda x : stemming(x))

print("Stemming Done :",round((time.time()-start)/60,2),"mins")

### COMBINE TOKENS ###
def join(teks):
    teks = " ".join([char for char in teks])
    return teks
df['Text_Preprocessed'] = df['stemming'].apply(lambda x : join(x))

### MEMISAH INPUT DAN OUTPUT ###
X = df.iloc[:,6] # TEKS HASIL PREPROCESSING / INPUT
y = df.iloc[:,1] # KELAS / OUTPUT


# def hapuss(data):   
#     if      data.find("(BE") == -1 \
#         and data.find("(Campuran; Disinformasi, Hasut, & Fakta)") == -1 \
#         and data.find("(DISINFORMASI") == -1 \
#         and data.find("(EDUKASI)") == -1 \
#         and data.find("(EVENTS)") == -1 \
#         and data.find("(F") == -1 \
#         and data.find("(H") == -1 \
#         and data.find("(ISU)") == -1 \
#         and data.find("(KLARIFIKASI)") == -1 \
#         and data.find("(MISINFORMASI)") == -1 \
#         and data.find("(SALAH)") == -1 \
#         and data.find("[A") == -1 \
#         and data.find("[B") == -1 \
#         and data.find("[C") == -1 \
#         and data.find("[D") == -1 \
#         and data.find("[E") == -1 \
#         and data.find("[F") == -1 \
#         and data.find("[H") == -1 \
#         and data.find("[I") == -1 \
#         and data.find("[K") == -1 \
#         and data.find("[M") == -1 \
#         and data.find("[R") == -1 \
#         and data.find("[S") == -1 \
#         and data.find("[T") == -1 \
#         and data.find("[U") == -1 \
#         and data.find("DISINFORMASI :") == -1 \
#         and data.find("EDUKASI :") == -1 \
#         and data.find("FITNAH :") == -1 \
#         and data.find("FITNAH:") == -1 \
#         and data.find("HASUT :") == -1 \
#         and data.find("HASUT:") == -1 \
#         and data.find("HOAX :") == -1 \
#         and data.find("HOAX:") == -1 \
#         and data.find("KLARIFIKASI :") == -1 \
#         and data.find("KLARIFIKASI :") == -1 \
#         and data.find("MISINFORMASI:") == -1 \
#         and data.find("SALAH]") == -1 : # data berbahasa inggris tidak dapat dijadikan input
#         data = data
#     else:
#         data = None
#     return data

# df4 = df4['title'].apply(lambda x : hapuss(x))

# def hl(data): # Hapus label pada judul
#     if  data.find("(BE") == -1 \
#         and data.find("(Campuran; Disinformasi, Hasut, & Fakta)") == -1 \
#         and data.find("(DISINFORMASI") == -1 \
#         and data.find("(EDUKASI)") == -1 \
#         and data.find("(EVENTS)") == -1 \
#         and data.find("(F") == -1 \
#         and data.find("(H") == -1 \
#         and data.find("(ISU)") == -1 \
#         and data.find("(KLARIFIKASI)") == -1 \
#         and data.find("(MISINFORMASI)") == -1 \
#         and data.find("(SALAH)") == -1 \
#         and data.find("[A") == -1 \
#         and data.find("[B") == -1 \
#         and data.find("[C") == -1 \
#         and data.find("[D") == -1 \
#         and data.find("[E") == -1 \
#         and data.find("[F") == -1 \
#         and data.find("[H") == -1 \
#         and data.find("[I") == -1 \
#         and data.find("[K") == -1 \
#         and data.find("[M") == -1 \
#         and data.find("[R") == -1 \
#         and data.find("[S") == -1 \
#         and data.find("[T") == -1 \
#         and data.find("[U") == -1 \
#         and data.find("DISINFORMASI :") == -1 \
#         and data.find("EDUKASI :") == -1 \
#         and data.find("FITNAH :") == -1 \
#         and data.find("FITNAH:") == -1 \
#         and data.find("HASUT :") == -1 \
#         and data.find("HASUT:") == -1 \
#         and data.find("HOAX :") == -1 \
#         and data.find("HOAX:") == -1 \
#         and data.find("KLARIFIKASI :") == -1 \
#         and data.find("KLARIFIKASI :") == -1 \
#         and data.find("MISINFORMASI:") == -1 \
#         and data.find("SALAH]") == -1 :
#         data = data
#     else:
#         data = data.replace(data.find("(BE"),"")
#         data = data.replace(data.find("(Campuran; Disinformasi, Hasut, & Fakta)"),"")
#         data = data.replace(data.find("(DISINFORMASI"),"")
#         data = data.replace(data.find("(EDUKASI)"),"")
#         data = data.replace(data.find("(EVENTS)"),"")
#         data = data.replace(data.find("(F"),"")
#         data = data.replace(data.find("(H"),"")
#         data = data.replace(data.find("(ISU)"),"")
#         data = data.replace(data.find("(KLARIFIKASI)"),"")
#         data = data.replace(data.find("(MISINFORMASI)"),"")
#         data = data.replace(data.find("(SALAH)"),"")
#         data = data.replace(data.find("[A"),"")
#         data = data.replace(data.find("[B"),"")
#         data = data.replace(data.find("[C"),"")
#         data = data.replace(data.find("[D"),"")
#         data = data.replace(data.find("[E"),"")
#         data = data.replace(data.find("[F"),"")
#         data = data.replace(data.find("[H"),"")
#         data = data.replace(data.find("[I"),"")
#         data = data.replace(data.find("[K"),"")
#         data = data.replace(data.find("[M"),"")
#         data = data.replace(data.find("[R"),"")
#         data = data.replace(data.find("[S"),"")
#         data = data.replace(data.find("[T"),"")
#         data = data.replace(data.find("[U"),"")
#         data = data.replace(data.find("DISINFORMASI :"),"")
#         data = data.replace(data.find("EDUKASI :"),"")
#         data = data.replace(data.find("FITNAH :"),"")
#         data = data.replace(data.find("FITNAH:"),"")
#         data = data.replace(data.find("HASUT :"),"")
#         data = data.replace(data.find("HASUT:"),"")
#         data = data.replace(data.find("HOAX :"),"")
#         data = data.replace(data.find("HOAX:"),"")
#         data = data.replace(data.find("KLARIFIKASI :"),"")
#         data = data.replace(data.find("KLARIFIKASI :"),"")
#         data = data.replace(data.find("MISINFORMASI:"),"")
#         data = data.replace(data.find("SALAH]"),"")
#     return data

# df4['title'] = df4['title'].apply(lambda x : hl(x))