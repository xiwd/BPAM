# BPAM: Recommendation Based on BP Neural Network with Attention Mechanism

This is our official implementation for the paper:

Wu-Dong Xi , Ling Huang , Chang-Dong Wang*, Yin-Yu Zheng, Jian-Huang Lai. [BPAM: Recommendation Based on BP Neural Network with Attention Mechanism](https://www.ijcai.org/proceedings/2019/542) BPAM: Recommendation Based on BP Neural Network with Attention Mechanism.  In IJCAI’19, Macao, China, August 10-16, 2019.


In this work, we propose a novel recommendation algorithm based on Back Propagation (BP) neural network with Attention Mechanism (BPAM). In particular, the BP neural network is utilized to learn the complex relationship of the target users and their neighbors. Compared with deep neural network, the shallow neural network, i.e., BP neural network, can not only reduce the computational and storage costs, but also prevent the model from over-fitting by the large number of parameters.In addition, an attention mechanism is designed to capture the global impact on all nearest target users for each user.

### Experiment

We are sorry, because our mistakes have led to errors in the results of the experiment, and hereby errata explains and corrects the results of the experiment.

In the code implementation, we used the library function *sklearn.neighbors.NearestNeighbors()* in python to get the user's k-nearest neighbors, but did not notice that the function's neighbors actually contain themselves, resulting in error. Due to type error, the prediction results of training sets in DMF is regarded as the prediction results of the test sets.We are very sorry for our carelessness again. Here we will update all the experimental results and upload the source code and the datasets of the algorithms for your verification. As we can see from the results, the effectiveness of our algorithm can still be guaranteed.


<p align='center'>
<img src= ''https://github.com/xiwd/BPAM/results/com.png'' title='images' style='max-width:800px'></img>
</p>

<p align='center'>
<img src=''https://github.com/xiwd/BPAM/results/var.png' title='images' style='max-width:600px'></img>
</p>

<p align='center'>
<img src=''https://github.com/xiwd/BPAM/results/par.png' title='images' style='max-width:800px'></img>
</p>

### Citation
    @inproceedings{BPAM19, 
      author= {Wu{-}Dong Xi and Ling Huang and Chang{-}Dong Wang and Yin{-}Yu Zheng andJianhuang Lai},
      title = {{BPAM:} Recommendation Based on {BP} Neural Network with Attention Mechanism},
      booktitle = {IJCAI'19},
      pages = {3905--3911},
      year  = {2019},
    }
