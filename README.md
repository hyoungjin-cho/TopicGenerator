# TopicGenerator

## Abstract
This report shows how LDA(Latent Dirichlet Allocation) Algorithm is applied to automatically generate tags for articles. MCMC methods like Gibb’s sampling are extensively applied when seeking for the tags that best represent the article. In general, this model returns the key words that best represent the material of the article by looking at the frequencies of these words are appearing.
## Dataset
The dataset used to test the LDA algorithm is obtained from the UCI Machine Learning Repository which is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms.It’s a dataset called “Sports articles for objectivity analysis Data set”, containing 1000 sports articles, and they were labeled by Amazon Mechanical Turk as objective or subjective. Among 1000 articles, I choose 5 articles with average 970 words to run through the algorithm. Due to the limitation of the computing power, I ran 250 iterations.
## Introduction to topic modeling
To deal with a large volume of text data, I am always looking for some topic or keyword generators, so that I can automatically label them and categorize them without fully reading the whole article manually. That’s where topic modeling comes into space. This algorithm is widely used in search engines, and they can categorize the articles and pop up the related academic reports or articles based on the user searches. Also, for recruiters who want to quickly overview a large volume of resumes, LDA can quickly extract each resume’s keywords and label them with the skill set and experiences. This enables recruiters quickly select the potential candidates by looking at the keywords instead of reading every resume.
The process of the topic modeling:
1. The first step of topic modeling is very similar to most natural languages. I start by data pre processing. This step includes removing all the punctuations, html tags and other non-material contents, converting all characters to lowercase, removing the stopwords in English. Finally convert the whole article to corpus.
2. After preprocessing the data, I assume every word could be the potential topic or key word of the article. Therefore, I will initialize the topic distribution (the distribution of each word to be topic) to a uniform distribution, and make each word as a state in the MCMC.
3. By running Gibb’s sampling, the MCMC works as a Markov chain starts shifting around the keywords by the frequencies they appear as well as the association with words nearby. For example, “computer” is associated with different words with different
probability, like a higher probability to come with “science”, but lower probability with “banana”. Therefore, if “science” and “banana” both appear in the dataset, it’s more likely to go to state “science” than state “banana”.
4. With this simulation process keep running, I’ll find that there are some states that’s more frequently visited than others. And this frequency can be considered as the posterior probability of each topic word, I can actually rank their posterior probability after the algorithm after sufficiently iterations, and the words with high posterior are most likely to be the topics.
## Theory Discussions:
LDA is a model that utilizes gibbs sampling and posterior probability updates to
do statistical inference for topic modeling. Given that the documents is a mix of multiple topics, I use LDA to learn the topic distribution in each document and also learn word distribution that is associated with each topic.
1. Set up parameters:
M denotes the number of documents
Ni is number of words in a given document i
α is the parameter of the Dirchlet prior on the per-document topic
distributions
β is the parameter of the Dirchlet prior on the per-topic word distributions πi is the topic distribution for the document i
Bk is the word distribution for topic k
Z(i,j) is the topic for the j-th word in document i
W(i,j) is the specific word
2. Generative process: πi ~ Dir(πi | α)
z(i,j) ~ Cat(z(i,j) | πi)
Bk ~ Dir(Bk | β)
w(i,j) ~ Cat(w(i, j) | z(i, j) =k, B)
3. Inference
The purpose of inference in LDA is that given a corpus, I can do statistical inference for finding underlying topics that explain the documents well.
P(πi | z(i,j) = k, Bk) = Dir(α + ∑_m I(z(i,m) = k))
P(Bk | z(i, j) = k, πi) = Dir(β + ∑_i∑_m I(w(i, m) = k))
 
 Pseudo code:
Intuition/simple examples:
  Now, suppose I have a set of documents above. I have chosen 4 fixed number of topics to discover(I can actually optimize over the number of topics to choose). I go through each document and randomly assign each word in the document to one of the K topics. After that, I will already have both the topic

 representations of all documents and word distributions of all the topics, not very good ones though. Then, I do updates using statistical inference.
Do this:
For each word w in d:
For each topic t:
I compute P(topic t | document d) and P(word w | topic
t). I reassign w a new topic with probability P(topic t | document d) * P(word w | topic t), which is the posterior probability. This is the probability that topic t generates word w.
So I keep updating for a large number of times, I will reach a steady state where the assignments are good to use.
Experiment Outcomes:
Here I display a portion of the snapshot of the outcomes, the complete outcome and procedures are given in the codes section.
Outcome:
​After running the LDA algorithm on the sample article, I obtained the result of topic modelling.
The outcome is generated after 250 iterations, and the topic words are arranged by frequency in descending order. As I can see from the screenshot, I have four words of “corpus” stick at the top, those are the words most likely to be the topics of this article. The prior distribution of all words are set to uniform, and I saw the impact of the algorithm made by the algorithm on this sampling process, as the posterior distribution was shifted from the prior, and converged to stationary distribution after a certain number of iterations.
   
Further Research:
Running the current algorithm on the sample dataset, I do see the LDA algorithm brought a posterior distribution of topic. As I manually read those articles, I found the article is closely related to the topics given by the output of the algorithm, which means the LDA method functioned properly. Looking further into this project, testing the accuracy of this algorithm will be an essential topic, as estimation of accuracy shouldn’t simply rely on human’s manual reading. And for sure, there are still far way to explore on how can I adjust and optimize the algorithm and how to test its accuracy and functionalities.
References:
[1] http://archive.ics.uci.edu/ml/datasets/Sports+articles+for+objectivity+analysis
[2] https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
[3] https://www.depends-on-the-definition.com/understanding-text-data-with-topic-models/
