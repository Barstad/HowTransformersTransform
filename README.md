# HOW DOES TRANSFORMERS TRANSFORM TEXT - REALLY?
Transformers are an excellent name, because that is literally what they are doing: transforming. It takes in a set of tokens and quite literally changes them by mixing them with other tokens in its context to propagate meaning across, and applying some fairly simple dense matrix operations. But, how does this look like? How does each token transform? If we track a token through the Transformer, how does it change? Can we interpret what the transformations are doing? 

In this app we explore how tokens transform by running them through the layers of a transformer. How does it look along the way?

To explore, we look at the tokens themselves, and their similarity between. We can select a token, and color code the other tokens based on the similarity to the selcted token. To get a sense of what the token means, we try to associate it with the base embeddings. Can we follow the meaning of the words? We would expect tokens to change meaning as it changes context. The embedding layer is trying to represent the tokens and embed them in a meaningful point in vector space. As we progress through the transformer, our representation changes. Can we follow how the context impacts the tokens? In the word cloud, we take the current selected layers representation of the selected token and try to map it back to the core embedding layer. Does the transformations make sense to us? We would expect that tokens change meaning somehow based on its context when we move into the layers. How does late layers look compared to early? This app lets us explore it.

https://github.com/user-attachments/assets/bda998be-463b-4778-83ca-1cf06c960457

  
