library(tm)
library(ggplot2)
library(wordcloud)
library(RWeka)
library(reshape2)
setwd('./Desktop/github/nlp_project')
source.cars = DirSource("./data/data_cars", encoding = "UTF-8")
corpus.cars = Corpus(source.cars)
length(corpus.cars)

myStopwords = c(stopwords(),"doc","favorite","datedate")

corpus.cars = sample (corpus.cars, size=500, replace =F)

tdm.cars = TermDocumentMatrix(corpus.cars)
tdm.cars

length(dimnames(tdm.cars)$Terms)
freq=rowSums(as.matrix(tdm.cars))
head(freq,10)
tail(freq,10)
plot(sort(freq, decreasing = T),col="blue",main="Word frequencies", xlab="Frequency-based rank", ylab = "Frequency")
tail(sort(freq),n=10)
sum(freq == 1)

tdm = TermDocumentMatrix(corpus.cars,
                         control=list(stopwords = T,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stemming = T))

tdm
length(dimnames(tdm)$Terms)
head(dimnames(tdm)$Terms,10)
tail(dimnames(tdm)$Terms,10)
freq=rowSums(as.matrix(tdm))
head(freq,10)
tail(freq,10)
plot(sort(freq, decreasing = T),col="blue",main="Word frequencies", xlab="Frequency-based rank", ylab = "Frequency")
tail(sort(freq),n=10)
sum(freq == 1)


tdm = TermDocumentMatrix(corpus.cars,
                         control=list(stopwords = myStopwords,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stemming = T))
tdm

freq=rowSums(as.matrix(tdm))
high.freq=tail(sort(freq),n=10)
hfp.df=as.data.frame(sort(high.freq))
hfp.df$names <- rownames(hfp.df) 
ggplot(hfp.df, aes(reorder(names,high.freq), high.freq)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Term frequencies")
tdm.tfidf = TermDocumentMatrix(corpus.cars,
                               control = list(weighting = weightTfIdf,
                                              stopwords = myStopwords, 
                                              removePunctuation = T,
                                              removeNumbers = T,
                                              stemming = T))
tdm.tfidf
freq=rowSums(as.matrix(tdm.tfidf))
plot(sort(freq, decreasing = T),col="blue",main="Word TF-IDF frequencies", xlab="TF-IDF-based rank", ylab = "TF-IDF")
tail(sort(freq),n=10)

asoc.car = as.data.frame(findAssocs(tdm,"car", 0.8))
asoc.car$names <- rownames(asoc.car) 
head(asoc.car)

ggplot(head(asoc.car,8), aes(reorder(names,car), car)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Correlation") +
  ggtitle("\"car\" associations")

asoc.seat = as.data.frame(findAssocs(tdm,"seat", 0.8))
asoc.seat$names <- rownames(asoc.seat) 
head(asoc.seat)

ggplot(head(asoc.seat,8), aes(reorder(names,seat), seat)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Correlation") +
  ggtitle("\"seat\" associations")

tdm.small = removeSparseTerms(tdm,0.05)
tdm.small
matrix.tdm = melt(as.matrix(tdm.small), value.name = "count")
head(matrix.tdm)
ggplot(matrix.tdm, aes(x = Docs, y = Terms, fill = log10(count))) +
  geom_tile(colour = "white") +
  scale_fill_gradient(high="#FF0000" , low="#FFFFFF")+
  ylab("Terms") +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  ggtitle("Word Doc Freq Graph, Cars")

pal=brewer.pal(8,"Blues")
pal=pal[-(1:3)]
set.seed(1234)
tdm.unigram = TermDocumentMatrix(corpus.cars,
                                 control=list(stopwords = c(myStopwords,"s","ve"),
                                              removePunctuation = T, 
                                              removeNumbers = T)) 

tdm.unigram = removeSparseTerms(tdm.unigram,0.8)
tdm.unigram

freq = sort(rowSums(as.matrix(tdm.unigram)), decreasing = T)
head(freq)
word.cloud=wordcloud(words=names(freq), freq=freq,
                     min.freq=5000, random.order=F, colors=pal)

corpus.ngrams = VCorpus(source.cars)
corpus.ngrams = sample (corpus.ngrams, size=500, replace =F)
corpus.ngrams = tm_map(corpus.ngrams,removeWords,c(myStopwords,"s","ve"))
corpus.ngrams = tm_map(corpus.ngrams,removePunctuation)
corpus.ngrams = tm_map(corpus.ngrams,removeNumbers)

Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 5, max = 5))
tdm.ngram = TermDocumentMatrix(corpus.ngrams,
                                control = list (tokenize = Tokenizer))

tdm.ngram = removeSparseTerms(tdm.ngram,0.99)
tdm.ngram

freq = sort(rowSums(as.matrix(tdm.ngram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent 5grams")


Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 6, max = 6))
tdm.ngram = TermDocumentMatrix(corpus.ngrams,
                               control = list (tokenize = Tokenizer))

tdm.ngram = removeSparseTerms(tdm.ngram,0.99)
tdm.ngram

freq = sort(rowSums(as.matrix(tdm.ngram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent 6grams")

Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 7, max = 7))
tdm.ngram = TermDocumentMatrix(corpus.ngrams,
                               control = list (tokenize = Tokenizer))

tdm.ngram = removeSparseTerms(tdm.ngram,0.99)
tdm.ngram

freq = sort(rowSums(as.matrix(tdm.ngram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent 7grams")
