# Image-Captionning

Dataset used for training, validation, and testing: Flickr 8k

M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and 
Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899


1. Introduction:

	The image caption task is one of the most popular research areas recently. There are several real world applications for this neuron network. For instance, with the help of image caption algorithm, visually impaired people will be much more capable of understanding the surroundings. Therefore, we used a neuron network algorithm that can take an image as the input and output a caption that can accurately describe the information contained in the image. Our proposed network is a combination of a convolutional neural network (CNN) to embed the information in the image and a recurrent neural network (RNN) to generate the caption of the image [1]. For CNN, we choose VGG11 to perform the embedding task and Long Short-term Memory recurrent neural network (LSTM) to perform the caption task. 

2. Illustration of the Model
![Image of model](https://user-images.githubusercontent.com/32416959/72279368-f6b3c600-3603-11ea-8267-e66e1cd3b53d.PNG)

Figure 1. the overview of the final model.

3. Related Work

	Image caption generation is a popular task in computer vision area. The achievement of this goal requires studies of machine learning for both image and language related tasks. There were similar studies of image caption posted in the last few years we found helpful. More specifically, this paper from 2015 [2], used a encoder-decoder framework similar to ours, with a CNN being the encoder and LSTM RNN model as decoder. But we use a different CNN model and different dataset. 

	Another paper from 2017 [3], proposed a more complex network which combines object detection, visual relationship detection and image caption network together to generate a higher accuracy and more natural description. This could be a potential to improve our model. 
	
4. Data Processing 

The dataset we used is Flickr8k[4]. It contains 8000 images manually selected  from the image hosting website, Flickr. Each image in the datasets have been labeled with five different captions describing the same image contents. These captions were used as ground truth for our image caption neural network. 

![Image](https://user-images.githubusercontent.com/32416959/72279840-f9fb8180-3604-11ea-859c-62c25cbc9617.PNG)
Figure 2, one sample data of the image and five captions for it.

In the dataset, 6000 images were used for training, and 1000 images for validation and test. Although the Flickr8k dataset have been manually selected for the purpose of including as many situations as possible, 8000 images were still not enough the cover all general living conditions. Especially considering images on flickr were posted by photographers, so the images were likely missing some regular living situations. 

Both the image and caption data need to be processed for out network. The images were formatted using Pytorch torchvision.transforms module. The image part will be input to a pretrained VGG convolutional neural network with an input size of 224*224. Therefore, the images were reshaped to the size 224*224, and converted to tensor with a normalization with mean as 0.5, and standard deviation as 0.5. So the each tensor element was in range between -1.0 to 1.0.

The captions were tokenized by words using nltk tokenizer. Words appearing less than a total of 5 times in the whole dataset were discarded to reduce the vocabulary size. The <start>, <end>, <unknown>, and <padding> tokens were placed. The vocabulary was embedded in an embedding vector dimension of 512, to be consistent with the VGG output size and LSTM RNN input size of 512.





5. Architecture

The  model is trying to generate as image caption after an image input. This is similar to a encoder-decoder framework as mentioned in the related work section. The input image first need to be encoded using a convolutional neural network to extract the feature from the image. And because the image will be given to a recurrent neural network, the feature output from CNN needs to be linear and has the same dimension as the input RNN. 

In our model, we used pretrained VGG-11 network as the CNN encoder. VGG-11 architecture is shown in figure on the right. A fully connected layer is added following VGG which inputs the image feature size 4096 and outputs the targeting embedding size 512. This image feature of size 512 will work as input to the RNN. In our model, we use LSTM shown below as the RNN, because  LSTM models have a good performance when dealing with language constructions.
![Image of model](https://user-images.githubusercontent.com/32416959/72279879-13043280-3605-11ea-9ce3-377b439a06c0.PNG)
![Image of model](https://user-images.githubusercontent.com/32416959/72280460-61fe9780-3606-11ea-9a21-10cb68ce1441.PNG)
Figure 3, sample figure for LSTM and VGG-11 [5]  network architecture.

The LSTM network has input size 512 and output size 512. The input is the image feature and the embedded caption word token sequence. The LSTM model is trained with teacher forcing, with the real caption between <start> and <end> token. 








6. Baseline Model
![Image of model](https://user-images.githubusercontent.com/32416959/72279942-37600f00-3605-11ea-8faa-4299091b9dbd.PNG)
Figure 4. Baseline model

The baseline model works like an auto-encoder: it first takes an image and send its embedding along with the caption ground truth to the decoder model, in our case, the Long short-term memory model (LSTM). The model first takes a resized image as the input and embeds the image with convolutional neural network, and pass the image features, the image embeddings, to the recurrent neural network. Since the model consists of two different parts, the CNN, convolutional neural network, and the RNN, recurrent neural network, we replace the CNN of our proposed model with Alexnet as the baseline model. The decoder part, LSTM model kept unchanged. Different from the proposed model with VGG11 as the image encoder, Alexnet has a simpler architecture consisting of convolutional layers, max pooling, dropout, data augmentation, ReLU activations and SGD. During our training with Alexnet, model trains with less time for each epoch. The comparison between the baseline model and the proposed model can be made by taking the same input image and observe the final result caption generated by each model. 

7. Quantitative Result

	The measurements we use to assess the performance of our neural network are loss values. We use cross entropy loss as the criterion to compute the loss value. The model first takes an image as the input and passes the resized image to a convolutional neural network. The image is embedded by the CNN layers and is passed to LSTM along with its corresponding captions. The loss value is computed by sending output of LSTM with its ground truth caption to the cross entropy loss function. The final loss values for the baseline model and the proposed model are 2.3192 and 2.3071 respectively after training for 5 epochs. Numerically, the proposed model out performs the baseline model with a lower loss function by training for a small number of epochs. In addition to the loss value, we could also implement Bilingual Evaluation Understudy score calculation to test the output accuracy by comparing the  outputs of the two model with the ground truth caption.
![Image of model](https://user-images.githubusercontent.com/32416959/72279943-37600f00-3605-11ea-87f1-6dc873f05341.PNG)
Figure 5. Mathematical Computation of Cross Entropy Loss Function [6]


8. Qualitative Result

	The model is trained for 5 epochs. Since we save the weights and loss value after each epoch into a pickle file, we can test the model using the weights and informations from each epoch.  Figure 6 and Figure 7 shows the results of running the model with the input image using weights from 5 epochs. After the first epoch, though the caption grammar is not perfect, the model can correctly detect the information from the image: for example, “man”, “red shirt”, and “bike”. After the second epoch, the model can generate a caption with correct information and accurate grammar. After the fourth and fifth epoch, the model detects information from the image with more accuracy and correct grammar. For example, in the caption from the fourth epoch, the model outputs results like “red jacket” and “helmet” instead of the “red shirt” during the first few trials. 

![Image of model](https://user-images.githubusercontent.com/32416959/72279944-37600f00-3605-11ea-89eb-b3f4eaeab4a2.PNG)
Figure 6. Input Image
![Image of model](https://user-images.githubusercontent.com/32416959/72279945-37600f00-3605-11ea-9bf1-d17e17d9e626.PNG)
Figure 7. Results of the Input Image using Weights from 5 Epochs

9. Discussion 

Overall, the model can detect most of the information from the input image and organizes words with correct grammar. Since we use Vgg11 as the convolutional layer, the image feature embeddings contains correct details and informations from the image. Besides, each image is trained for 5 times based on its 5 corresponding captions. Since the dataset we have for training describes each images with 5 different wordings, the model can learn how to organize the embedding features into correct sentences. The model can accurately detect colours of the object and its main features. Figure 8 and Figure 9 shows that the models gives correct colours of the shirt and the dog. However, when dealing with a simple object with no other figures or actions associated with the object, the model performs poorly; because the training dataset we use to feed the model include some figure and an action associated with the figure for the most cases. Figure 10 shows the case when the model does not perform well. Thus, the model can detect some person or animal and what he/she/it is doing in the image but not an object by its own. 
![Image of model](https://user-images.githubusercontent.com/32416959/72279896-20b9b800-3605-11ea-80a9-ba28a19549ae.PNG)
Figure 8. A Good Example Result from the Model
![Image of model](https://user-images.githubusercontent.com/32416959/72279941-37600f00-3605-11ea-9275-862a165a60ba.PNG)
Figure 9. Another Good Example Result Generated by the Model

![Image of model](https://user-images.githubusercontent.com/32416959/72279958-3c24c300-3605-11ea-843f-4c6cadbb1c78.PNG)
Figure 10. An Example when the Model Performed Poorly

10. Ethical Consideration 

	Since the proposed neuron network is designed to complete image recognition task, the misinterpretation of an image can cause serious ethical issue. In 2015, a software engineer in Google pointed out that the image recognition algorithm classified his black friend as “Gorilla”. [7]This was not respectful and fair for certain groups of people. So far, the dataset we used including Flickr 8k and Microsoft COCO are not proven to have demographic parity. Therefore, we should be careful about the accuracy of our proposed image caption neuron network. To prevent such a situation from happening, the dataset should achieve demographic parity to ensure every ethnic group have enough representation in the dataset.  

11. Project Difficulty

	The original proposal is to use the combination of Faster RCNN, VGG and LSTM to perform the image caption task. Besides only using VGG 11 to embed the information in the image, Faster RCNN, which can extract the information of the types of the objects presented in the image, their locations and the boundary boxes,  will provide extra inputs for LSTM. With more information provided, we expected that LSTM will perform better in the image caption task. However, since Faster RCNN requires long time to be trained even with the powerful GPU on Google. The estimation time to train Faster RCNN using Visual Genome (Containing 10,000 images) is 1 week if only one GPU is used. We did not have enough computational power to train the network. After we realized this problem, there was only 10 days left for the project. Therefore, we chose to use a pretrain Faster RCNN model trained using Microsoft COCO dataset. Since the dataset we used (Flickr 8k) has 6000 training set images and the time it takes for Faster RCNN around 5 seconds to extract the features from each image, total additional time for the model to train compared with the model which only has VGG and LSTM would be around 30,000 seconds (8.3 hours) per epoch. We noticed that we did not have enough time to tune the hyperparameters if we add Faster RCNN. Thus, we decided to remove Faster RCNN from our proposed model. With only VGG and LSTM, the training time per epoch is 15 minutes. We, thereby, had enough time to tune the hyperparameters to let the model perform better. This is the reason why our final model is 
differ than what we proposed at the beginning.



Reference:
[1]"Captioning Images with CNN and RNN, using PyTorch", Medium, 2019. [Online]. Available: https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3. [Accessed: 15- Aug- 2019].

[2]“Department of Computer Science,” Illinois. [Online]. Available: https://forms.illinois.edu/sec/1713398. [Accessed: 15-Aug-2019].
[3]Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan,”Show and Tell: A Neural Image Caption Generator” arXiv:1411.4555v2[cs.CV]. [Accessed: 15-Aug-2019]
[4]Yikang Li, Wanli Ouyang, BoleiZhou, Kun Wang, Xiaogang Wang, “Scene Graph Generation from Objects, Phrases and Region Captions“ arXiv:1707.09700v2[cs.CV]. [Accessed: 15-Aug-2019]

[5]Vladimir Iglovikov, Alexey Shvets, “TernausNet: U_Net with VGG11 Encoder Pre_trained on ImageNet for Image Segmentation” arXiv:1801.05746[cs.CV]. [Accessed: 15-Aug-2019]

[6]DiPietro, R. (2019). A Friendly Introduction to Cross-Entropy Loss. [online] Rdipietro.github.io. Available at: https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/ [Accessed 15 Aug. 2019].

[7]"Google ‘fixed’ its racist algorithm by removing gorillas from its image-labeling tech", The Verge, 2019. [Online]. Available: https://www.theverge.com/2018/1/12/16882408/google-racist-gorillas-photo-recognition-algorithm-ai. [Accessed: 15- Aug- 2019].
